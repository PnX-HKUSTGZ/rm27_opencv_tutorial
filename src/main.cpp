#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct LightBar {
    cv::RotatedRect rectangle;
    cv::Point2f top;
    cv::Point2f bottom;
    float length = 0.0F;
    float angle_degrees = 0.0F;
};

struct Armor {
    std::size_t left = 0;
    std::size_t right = 0;
    float score = 0.0F;
    // 顺序固定为左上、右上、右下、左下，方便后续绘制或姿态解算。
    std::array<cv::Point2f, 4> corners;
};

struct FrameResult {
    std::array<cv::Mat, 10> stages;
    cv::Mat annotated;
    std::size_t armor_count = 0;
};

cv::Point pixelPoint(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rectangle,
                     const cv::Scalar& color, int thickness = 2) {
    cv::Point2f vertices[4];
    rectangle.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(image, pixelPoint(vertices[i]), pixelPoint(vertices[(i + 1) % 4]),
                 color, thickness, cv::LINE_AA);
    }
}

// 第七步：用旋转矩形的长边表示灯条方向。直接使用 RotatedRect 的 angle
// 容易混淆长宽交换后的角度，所以从四个顶点计算长轴的上下端点。
bool makeLightBar(const std::vector<cv::Point>& contour, LightBar& bar) {
    if (cv::contourArea(contour) < 15.0) {
        return false;
    }

    const cv::RotatedRect rectangle = cv::minAreaRect(contour);
    const float length = std::max(rectangle.size.width, rectangle.size.height);
    const float width = std::min(rectangle.size.width, rectangle.size.height);
    if (width <= 0.0F || length < 18.0F || length / width < 2.5F) {
        return false;
    }

    cv::Point2f vertices[4];
    rectangle.points(vertices);
    cv::Point2f longest_edge;
    float longest_squared = -1.0F;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f edge = vertices[(i + 1) % 4] - vertices[i];
        const float squared = edge.dot(edge);
        if (squared > longest_squared) {
            longest_squared = squared;
            longest_edge = edge;
        }
    }
    if (longest_edge.y < 0.0F) {
        longest_edge *= -1.0F;
    }
    const cv::Point2f axis = longest_edge * (1.0F / std::sqrt(longest_squared));
    const float angle_degrees = std::atan2(axis.x, axis.y) * 180.0F / CV_PI;
    if (std::abs(angle_degrees) > 30.0F) {
        return false;
    }

    bar.rectangle = rectangle;
    bar.top = rectangle.center - axis * (length / 2.0F);
    bar.bottom = rectangle.center + axis * (length / 2.0F);
    bar.length = length;
    bar.angle_degrees = angle_degrees;
    return true;
}

// 第八步：灯条配对。所有长度和间距都用平均灯条长度归一化，
// 因而目标在画面里变大、变小时，阈值仍有相同的几何意义。
std::vector<Armor> pairLightBars(const std::vector<LightBar>& bars) {
    std::vector<Armor> candidates;
    for (std::size_t left = 0; left < bars.size(); ++left) {
        for (std::size_t right = left + 1; right < bars.size(); ++right) {
            const LightBar& a = bars[left];
            const LightBar& b = bars[right];
            const float mean_length = (a.length + b.length) / 2.0F;
            const float length_ratio = std::max(a.length, b.length) /
                                       std::min(a.length, b.length);
            const float gap_ratio = (b.rectangle.center.x - a.rectangle.center.x) /
                                    mean_length;
            const float height_difference =
                std::abs(a.rectangle.center.y - b.rectangle.center.y) / mean_length;
            const float angle_difference =
                std::abs(a.angle_degrees - b.angle_degrees);

            if (length_ratio > 1.7F || gap_ratio < 1.0F || gap_ratio > 3.0F ||
                height_difference > 0.5F || angle_difference > 20.0F) {
                continue;
            }

            Armor armor;
            armor.left = left;
            armor.right = right;
            armor.score = std::abs(gap_ratio - 2.0F) +
                          0.8F * height_difference +
                          0.6F * (length_ratio - 1.0F) +
                          0.03F * angle_difference;
            armor.corners = {a.top, b.top, b.bottom, a.bottom};
            candidates.push_back(armor);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Armor& a, const Armor& b) {
        return a.score < b.score;
    });

    // 同一根左灯条连接多个右灯条，或同一根右灯条连接多个左灯条时，
    // 这些候选区域会套叠，只保留几何分数较好的一个。相邻装甲板可以
    // 共用边界灯条：它在一组中是右灯条，在另一组中是左灯条。
    std::vector<bool> used_as_left(bars.size(), false);
    std::vector<bool> used_as_right(bars.size(), false);
    std::vector<Armor> selected;
    for (const Armor& armor : candidates) {
        if (used_as_left[armor.left] || used_as_right[armor.right]) {
            continue;
        }
        selected.push_back(armor);
        used_as_left[armor.left] = true;
        used_as_right[armor.right] = true;
    }
    return selected;
}

FrameResult processFrame(const cv::Mat& frame) {
    FrameResult result;
    // 第零步：BGR 原图。OpenCV 读取的视频帧默认按蓝、绿、红排列；
    // 保留一份原图，方便与后续每个处理阶段直接对照。
    result.stages[0] = frame.clone();

    // 第一步：滤波去噪。中值滤波用邻域的中间值替换孤立噪点，
    // 3×3 窗口较小，能尽量保留狭窄的蓝色灯条。
    cv::medianBlur(frame, result.stages[1], 3);

    // 第二步：亮度与颜色增强。先转到 HSV，把亮度 V 单独用 CLAHE
    // 做局部对比度增强；H（色相）和 S（饱和度）保持原样，便于找蓝色。
    cv::Mat hsv;
    cv::cvtColor(result.stages[1], hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(hsv_channels[2], hsv_channels[2]);
    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, result.stages[2], cv::COLOR_HSV2BGR);

    // 第三步：二值化。色相、饱和度和增强后的亮度先圈出蓝色；
    // 再要求 B 比 R/G 至少亮 20，避免暗背景或灰白数字进入掩膜。
    cv::Mat hsv_mask;
    cv::inRange(hsv, cv::Scalar(90, 65, 120), cv::Scalar(135, 255, 255), hsv_mask);
    std::vector<cv::Mat> bgr_channels;
    cv::split(result.stages[1], bgr_channels);
    cv::Mat max_red_green;
    cv::max(bgr_channels[1], bgr_channels[2], max_red_green);
    cv::Mat blue_contrast;
    cv::subtract(bgr_channels[0], max_red_green, blue_contrast);
    cv::Mat contrast_mask;
    cv::threshold(blue_contrast, contrast_mask, 19, 255, cv::THRESH_BINARY);
    cv::bitwise_and(hsv_mask, contrast_mask, result.stages[3]);

    // 第四步：形态学处理。竖向闭运算连接灯条中的小断点；
    // 随后的开运算删掉零散小亮点。核不能太大，否则相邻灯条会粘连。
    cv::morphologyEx(result.stages[3], result.stages[4], cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5)));
    cv::morphologyEx(result.stages[4], result.stages[4], cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // 第五步：轮廓提取。每个白色连通区域给出一条外轮廓；
    // 使用掩膜副本，保留原掩膜用于阶段图。
    cv::Mat contour_input = result.stages[4].clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_input, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    result.stages[5] = frame.clone();
    cv::drawContours(result.stages[5], contours, -1, cv::Scalar(255, 255, 0), 1);

    // 第六步：旋转矩形拟合。它同时描述区域的中心、长宽和方向，
    // 比水平包围框更适合略微倾斜的灯条。
    result.stages[6] = frame.clone();
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= 15.0) {
            drawRotatedRect(result.stages[6], cv::minAreaRect(contour),
                            cv::Scalar(255, 255, 0), 1);
        }
    }

    // 第七步：灯条几何筛选。只留下足够长、足够细、接近竖直的区域，
    // 并按照画面中的横坐标排序，为下一步左右配对做准备。
    std::vector<LightBar> bars;
    result.stages[7] = frame.clone();
    for (const auto& contour : contours) {
        LightBar bar;
        if (makeLightBar(contour, bar)) {
            bars.push_back(bar);
        }
    }
    std::sort(bars.begin(), bars.end(), [](const LightBar& a, const LightBar& b) {
        return a.rectangle.center.x < b.rectangle.center.x;
    });
    for (const LightBar& bar : bars) {
        drawRotatedRect(result.stages[7], bar.rectangle, cv::Scalar(0, 255, 0));
    }

    // 第八步：配对灯条。黄色线段连接通过长度、角度、位置检查的左右灯条。
    const std::vector<Armor> armors = pairLightBars(bars);
    result.stages[8] = result.stages[7].clone();
    for (const Armor& armor : armors) {
        cv::line(result.stages[8], pixelPoint(bars[armor.left].rectangle.center),
                 pixelPoint(bars[armor.right].rectangle.center),
                 cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // 第九步：生成装甲板四角点。左右灯条长轴的上下端点给出
    // 近似四角；红点和绿色四边形是最终输出，不代表精确的三维角点。
    result.annotated = frame.clone();
    for (std::size_t i = 0; i < armors.size(); ++i) {
        const Armor& armor = armors[i];
        for (int k = 0; k < 4; ++k) {
            cv::line(result.annotated, pixelPoint(armor.corners[k]),
                     pixelPoint(armor.corners[(k + 1) % 4]),
                     cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::circle(result.annotated, pixelPoint(armor.corners[k]), 4,
                       cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        }
        const cv::Point label = pixelPoint(armor.corners[0]) + cv::Point(0, -8);
        cv::putText(result.annotated, "Armor " + std::to_string(i + 1), label,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
    }
    result.stages[9] = result.annotated;
    result.armor_count = armors.size();
    return result;
}

// 教学阶段图采用英文短标签，因为 OpenCV 自带的 putText 不支持中文字符；
// 每一步的中文说明放在上面的源码注释和 README 中。
cv::Mat makeStageImage(const FrameResult& result) {
    constexpr int panel_width = 320;
    constexpr int panel_height = 180;
    constexpr std::array<const char*, 10> labels = {
        "BGR", "Denoise", "Enhance", "Binary", "Morphology",
        "Contours", "Rotated rectangles", "Light bars", "Pairs", "Armor corners"};
    cv::Mat montage(2 * panel_height, 5 * panel_width, CV_8UC3,
                    cv::Scalar(0, 0, 0));
    for (std::size_t i = 0; i < result.stages.size(); ++i) {
        cv::Mat color;
        if (result.stages[i].channels() == 1) {
            cv::cvtColor(result.stages[i], color, cv::COLOR_GRAY2BGR);
        } else {
            color = result.stages[i];
        }
        cv::Mat panel;
        cv::resize(color, panel, cv::Size(panel_width, panel_height));
        cv::rectangle(panel, cv::Rect(0, 0, panel_width, 26),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(panel, labels[i], cv::Point(8, 19),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
        const cv::Rect place(static_cast<int>(i % 5) * panel_width,
                             static_cast<int>(i / 5) * panel_height,
                             panel_width, panel_height);
        panel.copyTo(montage(place));
    }
    return montage;
}

void saveStageImage(const FrameResult& result, const fs::path& directory,
                    int frame_index) {
    std::ostringstream filename;
    filename << "frame_" << std::setw(6) << std::setfill('0') << frame_index
             << ".png";
    const fs::path path = directory / filename.str();
    if (!cv::imwrite(path.string(), makeStageImage(result))) {
        throw std::runtime_error("无法保存阶段图: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <输入视频.mp4> <输出目录>\n";
        return 1;
    }

    try {
        const fs::path input_path(argv[1]);
        const fs::path output_directory(argv[2]);
        if (!fs::is_regular_file(input_path)) {
            throw std::runtime_error("输入视频不存在: " + input_path.string());
        }

        cv::VideoCapture capture(input_path.string());
        if (!capture.isOpened()) {
            throw std::runtime_error("无法读取输入视频: " + input_path.string());
        }
        fs::create_directories(output_directory);
        const std::string stem = input_path.stem().string();
        const fs::path video_path = output_directory / (stem + "_annotated.mp4");
        const fs::path stage_directory = output_directory / (stem + "_steps");
        fs::create_directories(stage_directory);

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            fps = 30.0;
        }
        cv::VideoWriter writer;
        cv::Mat frame;
        FrameResult last_result;
        int frame_index = 0;
        std::size_t total_armors = 0;
        while (capture.read(frame)) {
            if (frame.empty()) {
                throw std::runtime_error("视频中出现空帧");
            }
            if (!writer.isOpened()) {
                writer.open(video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps, frame.size(), true);
                if (!writer.isOpened()) {
                    throw std::runtime_error("无法创建标注视频: " + video_path.string());
                }
            }

            FrameResult result = processFrame(frame);
            writer.write(result.annotated);
            total_armors += result.armor_count;
            if (frame_index % 30 == 0) {
                saveStageImage(result, stage_directory, frame_index);
            }
            last_result = std::move(result);
            ++frame_index;
        }
        if (frame_index == 0) {
            throw std::runtime_error("视频没有可读取的帧");
        }
        if ((frame_index - 1) % 30 != 0) {
            saveStageImage(last_result, stage_directory, frame_index - 1);
        }
        writer.release();
        capture.release();
        std::cout << input_path.filename().string() << ": " << frame_index
                  << " 帧，检测到 " << total_armors << " 组装甲板（逐帧计数）\n"
                  << "标注视频: " << video_path << "\n"
                  << "阶段图目录: " << stage_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
