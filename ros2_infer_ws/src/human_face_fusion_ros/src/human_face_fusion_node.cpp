/**
 * 人体（ScenePerceptionResult）+ 人脸（SCRFD）同图融合。
 * 新增：IoU 追踪器 / 自动注册 / 多向量 SQLite 人脸库 / 性别检测 / UpdatePersonName 服务。
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <ft2build.h>
#include FT_FREETYPE_H
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "remote_infer_bridge_cpp/msg/person_perception.hpp"
#include "remote_infer_bridge_cpp/msg/scene_perception_result.hpp"
#include "remote_infer_bridge_cpp/srv/update_person_name.hpp"
#include "scrfd_trt.h"
#include "sixdrepnet_trt.h"
#include "arcface_trt.h"
#include "face_database.h"
#include "gender_trt.h"
#include "iou_tracker.h"

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isJpegInImageMsg(const std::string& enc) {
    const std::string e = toLower(enc);
    return e.find("jpeg") != std::string::npos || e.find("jpg") != std::string::npos ||
           e.find("mjpeg") != std::string::npos || e.find("mjpg") != std::string::npos;
}

void ensureBgrU8C3(cv::Mat& img) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) { img.release(); return; }
    try {
        if (img.depth() != CV_8U) { cv::Mat u8; img.convertTo(u8, CV_8U); img = std::move(u8); }
        if (img.channels() == 3) return;
        cv::Mat c3;
        if (img.channels() == 1)      cv::cvtColor(img, c3, cv::COLOR_GRAY2BGR);
        else if (img.channels() == 4) cv::cvtColor(img, c3, cv::COLOR_BGRA2BGR);
        else { img.release(); return; }
        img = std::move(c3);
    } catch (const cv::Exception&) { img.release(); }
}

bool decodeColorImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg, cv::Mat& img) {
    const bool jpeg_magic = msg->data.size() >= 2 &&
                            static_cast<unsigned char>(msg->data[0]) == 0xff &&
                            static_cast<unsigned char>(msg->data[1]) == 0xd8;
    if (isJpegInImageMsg(msg->encoding) || jpeg_magic) {
        if (msg->data.empty()) return false;
        const int n = static_cast<int>(msg->data.size());
        cv::Mat jwrap(1, n, CV_8UC1, const_cast<unsigned char*>(msg->data.data()));
        img = cv::imdecode(jwrap, cv::IMREAD_COLOR);
        return !img.empty();
    }
    if (msg->width == 0 || msg->height == 0 || msg->step == 0) return false;
    const size_t expected = static_cast<size_t>(msg->step) * static_cast<size_t>(msg->height);
    if (msg->data.size() < expected) return false;
    const std::string encL = toLower(msg->encoding);
    const bool preferRgb = encL.find("rgb") != std::string::npos && encL.find("bgr") == std::string::npos;
    bool decoded = false;
    if (preferRgb) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
            cv::cvtColor(cv_ptr->image, img, cv::COLOR_RGB2BGR);
            decoded = !img.empty();
        } catch (...) {}
    }
    if (!decoded) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            img = cv_ptr->image; decoded = !img.empty();
        } catch (...) {}
    }
    if (!decoded) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
            cv::cvtColor(cv_ptr->image, img, cv::COLOR_RGB2BGR);
            decoded = !img.empty();
        } catch (...) {}
    }
    if (!decoded) {
        cv::Mat jwrap(1, static_cast<int>(msg->data.size()), CV_8UC1,
                      const_cast<unsigned char*>(msg->data.data()));
        img = cv::imdecode(jwrap, cv::IMREAD_COLOR);
        if (img.empty()) return false;
    }
    return true;
}

// 人脸纹理评分（Laplacian 方差）：打码/强模糊时会非常低。
float faceTextureVariance(const cv::Mat& bgr_face) {
    if (bgr_face.empty()) return 0.f;
    cv::Mat gray;
    if (bgr_face.channels() == 3) cv::cvtColor(bgr_face, gray, cv::COLOR_BGR2GRAY);
    else gray = bgr_face;
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F, 3);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return static_cast<float>(stddev[0] * stddev[0]);
}

rclcpp::QoS makeColorSubQoS(const std::string& mode) {
    std::string m = mode;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (m == "default" || m == "reliable") {
        rclcpp::QoS q(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default), rmw_qos_profile_default);
        q.keep_last(8);
        return q;
    }
    return rclcpp::SensorDataQoS().keep_last(15);
}

// ── Visualization constants ───────────────────────────────────────────────────
constexpr int kFont = cv::FONT_HERSHEY_DUPLEX;
const cv::Scalar kBodyOuter(200, 160, 0);
const cv::Scalar kBodyInner(255, 255, 0);
const cv::Scalar kFaceOuter(160, 0, 200);
const cv::Scalar kFaceInner(255, 80, 255);
const cv::Scalar kLabelBg(22, 24, 28);
const cv::Scalar kOutline(8, 8, 10);
static constexpr const char* kCjkFont =
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";

inline bool hasNonAscii(const char* s) {
    for (; *s; ++s) if (static_cast<unsigned char>(*s) > 127) return true;
    return false;
}

static uint32_t utf8Next(const uint8_t*& p) {
    uint32_t cp;
    if      ((*p & 0x80) == 0x00) { cp = *p++; }
    else if ((*p & 0xE0) == 0xC0) { cp  = (*p++ & 0x1F) << 6;  cp |= (*p++ & 0x3F); }
    else if ((*p & 0xF0) == 0xE0) { cp  = (*p++ & 0x0F) << 12; cp |= (*p++ & 0x3F) << 6;  cp |= (*p++ & 0x3F); }
    else                           { cp  = (*p++ & 0x07) << 18; cp |= (*p++ & 0x3F) << 12; cp |= (*p++ & 0x3F) << 6; cp |= (*p++ & 0x3F); }
    return cp;
}

struct FT2Text {
    FT_Library lib{};
    FT_Face    face{};
    bool       ok{false};
    FT2Text() {
        if (FT_Init_FreeType(&lib)) return;
        if (FT_New_Face(lib, kCjkFont, 0, &face)) { FT_Done_FreeType(lib); return; }
        ok = true;
    }
    ~FT2Text() { if (ok) { FT_Done_Face(face); FT_Done_FreeType(lib); } }
    cv::Size size(const char* text, int fh, int* bl) {
        if (!ok) return {};
        FT_Set_Pixel_Sizes(face, 0, static_cast<FT_UInt>(fh));
        const int asc  =  face->size->metrics.ascender  >> 6;
        const int desc = -face->size->metrics.descender >> 6;
        if (bl) *bl = desc;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(text);
        int w = 0;
        while (*p) { uint32_t cp = utf8Next(p); if (!FT_Load_Char(face, cp, FT_LOAD_DEFAULT)) w += face->glyph->advance.x >> 6; }
        return {w, asc};
    }
    void draw(cv::Mat& img, const char* text, cv::Point org, int fh, const cv::Scalar& col) {
        if (!ok || img.empty() || img.type() != CV_8UC3) return;
        FT_Set_Pixel_Sizes(face, 0, static_cast<FT_UInt>(fh));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(text);
        int penX = org.x;
        while (*p) {
            uint32_t cp = utf8Next(p);
            if (FT_Load_Char(face, cp, FT_LOAD_RENDER)) continue;
            const FT_GlyphSlot g = face->glyph;
            const FT_Bitmap& bmp = g->bitmap;
            const int bx = penX + g->bitmap_left, by = org.y - g->bitmap_top;
            for (int r = 0; r < static_cast<int>(bmp.rows); ++r) {
                const int iy = by + r; if (iy < 0 || iy >= img.rows) continue;
                uint8_t* row = img.ptr<uint8_t>(iy);
                for (int c = 0; c < static_cast<int>(bmp.width); ++c) {
                    const int ix = bx + c; if (ix < 0 || ix >= img.cols) continue;
                    const float a = bmp.buffer[r * bmp.pitch + c] / 255.f;
                    if (a < 0.01f) continue;
                    const float ia = 1.f - a;
                    row[ix*3+0] = static_cast<uint8_t>(row[ix*3+0] * ia + col[0] * a);
                    row[ix*3+1] = static_cast<uint8_t>(row[ix*3+1] * ia + col[1] * a);
                    row[ix*3+2] = static_cast<uint8_t>(row[ix*3+2] * ia + col[2] * a);
                }
            }
            penX += g->advance.x >> 6;
        }
    }
};

static FT2Text& ft2() { static FT2Text inst; return inst; }
inline int ft2Height(double s) { return std::max(8, static_cast<int>(s * 30 + 0.5)); }

void putTextOutlined(cv::Mat& img, const char* text, cv::Point org, double scale,
                     const cv::Scalar& fg, int thick) {
    if (hasNonAscii(text)) {
        const int fh = ft2Height(scale);
        static const int kO[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (const auto& d : kO) ft2().draw(img, text, {org.x+d[0], org.y+d[1]}, fh, kOutline);
        ft2().draw(img, text, org, fh, fg);
        return;
    }
    static const int kO[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (const auto& d : kO)
        cv::putText(img, text, {org.x+d[0], org.y+d[1]}, kFont, scale, kOutline, thick+1, cv::LINE_AA);
    cv::putText(img, text, org, kFont, scale, fg, thick, cv::LINE_AA);
}

void drawCornerBrackets(cv::Mat& bgr, const cv::Rect& r, const cv::Scalar& col, int len, int thick) {
    const int x0=r.x, y0=r.y, x1=r.x+r.width-1, y1=r.y+r.height-1;
    len = std::min(len, std::max(4, std::min(r.width, r.height)/4));
    cv::line(bgr,{x0,y0},{x0+len,y0},col,thick,cv::LINE_AA); cv::line(bgr,{x0,y0},{x0,y0+len},col,thick,cv::LINE_AA);
    cv::line(bgr,{x1,y0},{x1-len,y0},col,thick,cv::LINE_AA); cv::line(bgr,{x1,y0},{x1,y0+len},col,thick,cv::LINE_AA);
    cv::line(bgr,{x0,y1},{x0+len,y1},col,thick,cv::LINE_AA); cv::line(bgr,{x0,y1},{x0,y1-len},col,thick,cv::LINE_AA);
    cv::line(bgr,{x1,y1},{x1-len,y1},col,thick,cv::LINE_AA); cv::line(bgr,{x1,y1},{x1,y1-len},col,thick,cv::LINE_AA);
}

cv::Rect inflateRect(cv::Rect r, int m, int W, int H) {
    r.x-=m; r.y-=m; r.width+=2*m; r.height+=2*m;
    return r & cv::Rect(0,0,W,H);
}

bool labelOverlapsOccupied(const cv::Rect& bg, const std::vector<cv::Rect>& occ) {
    for (const cv::Rect& o : occ) if ((bg & o).area() > 0) return true;
    return false;
}

void drawTagLabelAvoid(cv::Mat& bgr, const cv::Rect& anchor, const char* line,
                       const cv::Scalar& accent, bool preferAbove, std::vector<cv::Rect>& occ) {
    int baseline = 0;
    const double kScale = 0.52; const int kThick = 1;
    cv::Size sz;
    if (hasNonAscii(line)) sz = ft2().size(line, ft2Height(kScale), &baseline);
    else sz = cv::getTextSize(line, kFont, kScale, kThick, &baseline);
    const int padX=7, padY=5;
    const int lw=sz.width+padX*2+6, lh=sz.height+baseline+padY*2;
    const int W=bgr.cols, H=bgr.rows;
    const int tyTop0 = anchor.y-lh, tyBot0 = anchor.y+anchor.height+4;
    struct Offset { int dx, dy; bool above; };
    std::vector<Offset> cand; cand.reserve(64);
    if (preferAbove) {
        for (int dy=0;dy>=-72;dy-=6){for(int dx=0;dx<=80;dx+=8)cand.push_back({dx,dy,true});for(int dx=-8;dx>=-80;dx-=8)cand.push_back({dx,dy,true});}
        for (int dy=6;dy<=72;dy+=6){for(int dx=0;dx<=80;dx+=8)cand.push_back({dx,dy,true});for(int dx=-8;dx>=-80;dx-=8)cand.push_back({dx,dy,true});}
        for (int dy=0;dy<=72;dy+=6){for(int dx=0;dx<=80;dx+=8)cand.push_back({dx,dy,false});for(int dx=-8;dx>=-80;dx-=8)cand.push_back({dx,dy,false});}
    } else {
        for (int dy=0;dy<=96;dy+=6){for(int dx=0;dx<=100;dx+=8)cand.push_back({dx,dy,false});for(int dx=-8;dx>=-100;dx-=8)cand.push_back({dx,dy,false});}
        for (int dy=-6;dy>=-72;dy-=6){for(int dx=0;dx<=80;dx+=8)cand.push_back({dx,dy,false});for(int dx=-8;dx>=-80;dx-=8)cand.push_back({dx,dy,false});}
        for (int dy=0;dy>=-72;dy-=6){for(int dx=0;dx<=80;dx+=8)cand.push_back({dx,dy,true});for(int dx=-8;dx>=-80;dx-=8)cand.push_back({dx,dy,true});}
    }
    for (const Offset& c : cand) {
        const int tx = std::clamp(anchor.x+c.dx, 0, std::max(0, W-lw));
        const int labelY = std::clamp(c.above?(tyTop0+c.dy):(tyBot0+c.dy), 0, std::max(0, H-lh));
        cv::Rect bg(tx, labelY, lw, lh); bg &= cv::Rect(0,0,W,H);
        if (bg.width < 4 || bg.height < 4) continue;
        if (labelOverlapsOccupied(bg, occ)) continue;
        cv::rectangle(bgr, bg, kLabelBg, -1);
        cv::rectangle(bgr, bg, accent, 1);
        cv::rectangle(bgr, cv::Rect(bg.x,bg.y,4,bg.height), accent, -1);
        const cv::Point textOrg(bg.x+padX+4, bg.y+bg.height-padY-2);
        putTextOutlined(bgr, line, textOrg, kScale, accent, kThick);
        occ.push_back(inflateRect(bg, 2, W, H));
        return;
    }
    // Fallback: draw at preferred position without overlap check
    {
        const int tx = std::clamp(anchor.x, 0, std::max(0, W-lw));
        const int labelY = std::clamp(preferAbove?tyTop0:tyBot0, 0, std::max(0, H-lh));
        cv::Rect bg(tx, labelY, lw, lh); bg &= cv::Rect(0,0,W,H);
        if (bg.width >= 4 && bg.height >= 4) {
            cv::rectangle(bgr, bg, kLabelBg, -1);
            cv::rectangle(bgr, bg, accent, 1);
            cv::rectangle(bgr, cv::Rect(bg.x,bg.y,4,bg.height), accent, -1);
            const cv::Point textOrg(bg.x+padX+4, bg.y+bg.height-padY-2);
            putTextOutlined(bgr, line, textOrg, kScale, accent, kThick);
            occ.push_back(inflateRect(bg, 2, W, H));
        }
    }
}

void drawFusionLegendCompact(cv::Mat& bgr) {
    if (bgr.rows < 64 || bgr.cols < 200) return;
    const char* t1="person + m"; const char* t2="face + %";
    int baseline=0; const double scale=0.42;
    const cv::Size s1=cv::getTextSize(t1,kFont,scale,1,&baseline);
    const cv::Size s2=cv::getTextSize(t2,kFont,scale,1,&baseline);
    const int padY=5, textH=std::max(s1.height+baseline,s2.height+baseline);
    const int barH=std::max(22,textH+padY*2), marginX=8, marginB=6;
    const int y0=bgr.rows-barH-marginB, w=bgr.cols-marginX*2;
    if (w<80||y0<0) return;
    cv::Rect bar(marginX,y0,w,barH);
    cv::rectangle(bgr,bar,kLabelBg,-1);
    cv::line(bgr,{bar.x,bar.y},{bar.x+bar.width,bar.y},kBodyInner,1,cv::LINE_AA);
    const int cy=bar.y+barH/2, textY=bar.y+barH-padY-2;
    cv::rectangle(bgr,{bar.x+8,cy-5},{bar.x+20,cy+5},kBodyInner,-1);
    cv::rectangle(bgr,{bar.x+8,cy-5},{bar.x+20,cy+5},kBodyOuter,1);
    putTextOutlined(bgr,t1,{bar.x+26,textY},scale,kBodyInner,1);
    const int xChip2=bar.x+26+s1.width+14;
    cv::rectangle(bgr,{xChip2,cy-5},{xChip2+12,cy+5},kFaceInner,-1);
    cv::rectangle(bgr,{xChip2,cy-5},{xChip2+12,cy+5},kFaceOuter,1);
    putTextOutlined(bgr,t2,{xChip2+18,textY},scale,kFaceInner,1);
}

void buildOccupiedFromDetections(const cv::Mat& bgr,
                                 const remote_infer_bridge_cpp::msg::ScenePerceptionResult* fr,
                                 bool use_dets, std::vector<cv::Rect>& occ) {
    occ.clear();
    if (!use_dets || fr == nullptr) return;
    const int W=bgr.cols, H=bgr.rows;
    for (const auto& p : fr->persons) {
        cv::Rect rb(std::max(0,p.body_x),std::max(0,p.body_y),std::max(0,p.body_w),std::max(0,p.body_h));
        rb &= cv::Rect(0,0,W,H);
        if (rb.width>=2&&rb.height>=2) occ.push_back(inflateRect(rb,4,W,H));
        if (p.has_face) {
            cv::Rect rf(std::max(0,p.face_x),std::max(0,p.face_y),std::max(0,p.face_w),std::max(0,p.face_h));
            rf &= cv::Rect(0,0,W,H);
            if (rf.width>=2&&rf.height>=2) occ.push_back(inflateRect(rf,3,W,H));
        }
    }
}

void drawPersonBoxes(cv::Mat& bgr, const remote_infer_bridge_cpp::msg::ScenePerceptionResult& fr) {
    for (const auto& p : fr.persons) {
        cv::Rect r(std::max(0,p.body_x),std::max(0,p.body_y),std::max(0,p.body_w),std::max(0,p.body_h));
        r &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if (r.width<2||r.height<2) continue;
        const int cl=std::clamp(std::min(r.width,r.height)/5,12,48);
        cv::rectangle(bgr,r,kBodyOuter,3);
        drawCornerBrackets(bgr,r,kBodyInner,cl,2);
    }
}

void drawPersonLabels(cv::Mat& bgr, const remote_infer_bridge_cpp::msg::ScenePerceptionResult& fr,
                      std::vector<cv::Rect>& occ) {
    for (const auto& p : fr.persons) {
        cv::Rect r(std::max(0,p.body_x),std::max(0,p.body_y),std::max(0,p.body_w),std::max(0,p.body_h));
        r &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if (r.width<2||r.height<2) continue;
        char text[128];
        std::snprintf(text, sizeof(text), "PERSON  %.2f m", static_cast<double>(p.distance));
        drawTagLabelAvoid(bgr, r, text, kBodyInner, true, occ);
    }
}

using PP = remote_infer_bridge_cpp::msg::PersonPerception;

void drawFaceBoxes(cv::Mat& bgr, const std::vector<PP>& persons) {
    for (const auto& p : persons) {
        if (!p.has_face) continue;
        cv::Rect r(std::max(0,p.face_x),std::max(0,p.face_y),std::max(0,p.face_w),std::max(0,p.face_h));
        r &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if (r.width<2||r.height<2) continue;
        const int cl=std::clamp(std::min(r.width,r.height)/4,10,40);
        cv::rectangle(bgr,r,kFaceOuter,2);
        drawCornerBrackets(bgr,r,kFaceInner,cl,2);
    }
}

// BGR: 蓝色(男) 、粉色(女)
const cv::Scalar kGenderMaleBg  (180, 80,  10);   // 深蓝
const cv::Scalar kGenderMaleFg  (255, 200, 120);  // 亮蓝白
const cv::Scalar kGenderFemaleBg(60,  40,  160);  // 深粉
const cv::Scalar kGenderFemaleFg(220, 160, 255);  // 亮粉

/** 在人脸框右上角外侧画性别徽标：♂ 蓝色 / ♀ 粉色 */
void drawGenderBadge(cv::Mat& bgr, const PP& p) {
    if (!p.has_face) return;
    if (p.gender == PP::GENDER_UNKNOWN) return;

    const cv::Rect face(std::max(0,p.face_x),std::max(0,p.face_y),
                        std::max(1,p.face_w),std::max(1,p.face_h));
    if (face.width < 10 || face.height < 10) return;

    const bool male = (p.gender == PP::GENDER_MALE);
    // UTF-8: ♂ = E2 99 82, ♀ = E2 99 80
    const char* sym  = male ? "\xe2\x99\x82" : "\xe2\x99\x80";
    // BGR colors
    const cv::Scalar bg = male ? cv::Scalar(160, 60, 10)  : cv::Scalar(80,  40, 180);
    const cv::Scalar fg = male ? cv::Scalar(255, 220, 180): cv::Scalar(255, 180, 255);
    const cv::Scalar border = male ? cv::Scalar(255,180,80): cv::Scalar(200,100,255);

    // Badge size: fixed comfortable size, min 26px height
    const int bh = std::clamp(face.height / 5, 26, 40);
    const int fh = bh - 4;
    const int bw = bh + 4;  // slightly wider for the symbol

    // Position: top-right corner of face box, just outside (above)
    const int W = bgr.cols, H = bgr.rows;
    const int bx = std::clamp(face.x + face.width - bw, 0, W - bw);
    const int by = std::clamp(face.y - bh - 2, 0, H - bh);

    cv::Rect badge(bx, by, bw, bh);
    badge &= cv::Rect(0, 0, W, H);
    if (badge.width < 8 || badge.height < 8) return;

    // Filled background with rounded look (thick border)
    cv::rectangle(bgr, badge, bg, -1);
    cv::rectangle(bgr, badge, border, 2);

    // Draw symbol centered in badge
    const int textX = badge.x + (badge.width - fh) / 2;
    const int textY = badge.y + badge.height - 3;
    if (ft2().ok) {
        ft2().draw(bgr, sym, {textX, textY}, fh, fg);
    } else {
        const char* ascii = male ? "M" : "F";
        cv::putText(bgr, ascii, {badge.x + 4, textY},
                    cv::FONT_HERSHEY_DUPLEX, 0.55, fg, 1, cv::LINE_AA);
    }
}

void drawFaceLabels(cv::Mat& bgr, const std::vector<PP>& persons, std::vector<cv::Rect>& occ) {
    std::vector<const PP*> order;
    for (const auto& p : persons) if (p.has_face) order.push_back(&p);
    std::sort(order.begin(), order.end(), [](const PP* a, const PP* b){ return a->face_x < b->face_x; });

    for (const PP* p : order) {
        cv::Rect r(std::max(0,p->face_x),std::max(0,p->face_y),std::max(0,p->face_w),std::max(0,p->face_h));
        r &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if (r.width<2||r.height<2) continue;

        char text[256];
        if (!p->person_name.empty()) {
            std::snprintf(text, sizeof(text), "%s  %.0f%%",
                          p->person_name.c_str(),
                          static_cast<double>(p->face_recog_conf) * 100.0);
        } else if (!p->person_uuid.empty()) {
            std::snprintf(text, sizeof(text), "ID:%.8s  %.0f%%",
                          p->person_uuid.c_str(),
                          static_cast<double>(p->face_recog_conf) * 100.0);
        } else {
            std::snprintf(text, sizeof(text), "FACE  %.0f%%",
                          static_cast<double>(p->face_conf) * 100.0);
        }
        drawTagLabelAvoid(bgr, r, text, kFaceInner, false, occ);
    }
}

float faceBoxIoU(const FaceObject& a, const FaceObject& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    const float ia = inter.area();
    if (ia <= 0.f) return 0.f;
    const float u = a.rect.area() + b.rect.area() - ia;
    return u > 1e-6f ? ia / u : 0.f;
}

void nmsFaceObjectsInPlace(std::vector<FaceObject>& faces, float nms_threshold) {
    if (faces.size() <= 1) return;
    std::sort(faces.begin(), faces.end(), [](const FaceObject& a, const FaceObject& b){ return a.prob > b.prob; });
    std::vector<int> picked;
    const int n = static_cast<int>(faces.size());
    for (int i = 0; i < n; ++i) {
        int keep = 1;
        for (int j : picked) if (faceBoxIoU(faces[i], faces[j]) > nms_threshold) keep = 0;
        if (keep) picked.push_back(i);
    }
    std::vector<FaceObject> out; out.reserve(picked.size());
    for (int idx : picked) out.push_back(std::move(faces[idx]));
    faces.swap(out);
}

void offsetFaceToGlobal(FaceObject& f, float dx, float dy) {
    f.rect.x += dx; f.rect.y += dy;
    for (int i = 0; i < 5; ++i) { f.landmark[i].x += dx; f.landmark[i].y += dy; }
}

void scalePersonsToImage(remote_infer_bridge_cpp::msg::ScenePerceptionResult& fr, int img_w, int img_h) {
    if (fr.image_width == 0u || fr.image_height == 0u || img_w <= 0 || img_h <= 0) return;
    if (fr.image_width == static_cast<uint32_t>(img_w) && fr.image_height == static_cast<uint32_t>(img_h)) return;
    const float sx = static_cast<float>(img_w) / static_cast<float>(fr.image_width);
    const float sy = static_cast<float>(img_h) / static_cast<float>(fr.image_height);
    for (auto& p : fr.persons) {
        p.body_x = static_cast<int32_t>(std::lround(static_cast<float>(p.body_x) * sx));
        p.body_y = static_cast<int32_t>(std::lround(static_cast<float>(p.body_y) * sy));
        p.body_w = static_cast<int32_t>(std::lround(static_cast<float>(p.body_w) * sx));
        p.body_h = static_cast<int32_t>(std::lround(static_cast<float>(p.body_h) * sy));
    }
    fr.image_width = static_cast<uint32_t>(img_w);
    fr.image_height = static_cast<uint32_t>(img_h);
}

cv::Rect personHeadCropRect(const remote_infer_bridge_cpp::msg::PersonPerception& p,
                             int W, int H,
                             double head_h_ratio, double width_pad_ratio, double top_expand_ratio) {
    const int x=std::max(0,p.body_x), y=std::max(0,p.body_y);
    const int w=std::max(0,p.body_w), h=std::max(0,p.body_h);
    if (w<4||h<4) return {};
    const double cx = static_cast<double>(x) + 0.5*static_cast<double>(w);
    const int roi_w = std::max(8, static_cast<int>(std::ceil(static_cast<double>(w)*(1.0+2.0*width_pad_ratio))));
    const int expand_up = std::min(y, static_cast<int>(std::ceil(static_cast<double>(h)*std::max(0.0,top_expand_ratio))));
    int roi_y = std::max(0, y-expand_up);
    if (roi_y >= H) return {};
    const int base_h = std::max(8, static_cast<int>(std::ceil(static_cast<double>(h)*head_h_ratio)));
    const int roi_h = std::max(1, std::min(H-roi_y, base_h+expand_up));
    int roi_x = static_cast<int>(std::floor(cx - 0.5*static_cast<double>(roi_w)));
    cv::Rect r(roi_x, roi_y, roi_w, roi_h);
    return r & cv::Rect(0,0,W,H);
}

cv::Rect expandFaceRect(const cv::Rect_<float>& r, float ratio, int W, int H) {
    const float cx = r.x+r.width*0.5f, cy = r.y+r.height*0.5f;
    const float nw = r.width*(1.f+2.f*ratio), nh = r.height*(1.f+2.f*ratio);
    cv::Rect ir(static_cast<int>(std::floor(cx-nw*0.5f)),
                static_cast<int>(std::floor(cy-nh*0.5f)),
                static_cast<int>(std::ceil(nw)),
                static_cast<int>(std::ceil(nh)));
    return ir & cv::Rect(0,0,W,H);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// HumanFaceFusionNode
// ─────────────────────────────────────────────────────────────────────────────
class HumanFaceFusionNode : public rclcpp::Node {
public:
    HumanFaceFusionNode() : Node("human_face_fusion_node") {
        // ── Existing parameters ───────────────────────────────────────────────
        engine_path_       = declare_parameter<std::string>("engine_path", "");
        image_topic_       = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
        detections_topic_  = declare_parameter<std::string>("detections_topic", "/remote_infer/detections");
        fused_topic_       = declare_parameter<std::string>("fused_topic", "/human_face_fusion/image_viz");
        publish_fused_     = declare_parameter<bool>("publish_fused", true);
        show_window_       = declare_parameter<bool>("show_window", true);
        window_name_       = declare_parameter<std::string>("window_name", "human_face_fusion");
        nms_threshold_     = static_cast<float>(declare_parameter<double>("nms_threshold", 0.4));
        detection_stale_ms_= declare_parameter<int>("detection_stale_ms", 500);
        person_head_height_ratio_      = declare_parameter<double>("person_head_height_ratio", 0.58);
        person_head_width_pad_ratio_   = declare_parameter<double>("person_head_width_pad_ratio", 0.24);
        person_head_top_expand_ratio_  = declare_parameter<double>("person_head_top_expand_ratio", 0.14);
        face_roi_min_side_             = declare_parameter<int>("face_roi_min_side", 48);
        face_max_person_rois_          = declare_parameter<int>("face_max_person_rois", 8);
        align_detection_dims_          = declare_parameter<bool>("align_detection_dims", true);
        face_persons_fallback_ms_      = declare_parameter<int>("face_persons_fallback_ms", 350);
        face_persons_fallback_ms_ = std::clamp(face_persons_fallback_ms_, 0, 5000);
        roi_scrfd_prob_threshold_ = static_cast<float>(declare_parameter<double>("roi_scrfd_prob_threshold", 0.38));
        roi_scrfd_prob_threshold_ = std::clamp(roi_scrfd_prob_threshold_, 0.08f, 0.95f);
        roi_scrfd_nms_threshold_  = static_cast<float>(declare_parameter<double>("roi_scrfd_nms_threshold", 0.45));
        roi_scrfd_nms_threshold_  = std::clamp(roi_scrfd_nms_threshold_, 0.15f, 0.95f);
        person_head_height_ratio_     = std::clamp(person_head_height_ratio_,    0.18, 0.78);
        person_head_width_pad_ratio_  = std::clamp(person_head_width_pad_ratio_, 0.0,  0.55);
        person_head_top_expand_ratio_ = std::clamp(person_head_top_expand_ratio_,0.0,  0.5);
        if (face_roi_min_side_ < 32)  face_roi_min_side_ = 32;
        if (face_max_person_rois_ < 1) face_max_person_rois_ = 1;
        color_sub_qos_mode_ = declare_parameter<std::string>("color_sub_qos", "sensor_data");
        also_subscribe_orbbec_composable_color_ =
            declare_parameter<bool>("also_subscribe_orbbec_composable_color", false);
        log_period_ms_             = declare_parameter<int>("log_period_ms", 2000);
        viz_show_legend_           = declare_parameter<bool>("viz_show_legend", false);
        log_frame_timing_          = declare_parameter<bool>("log_frame_timing", false);
        log_frame_timing_period_ms_= declare_parameter<int>("log_frame_timing_period_ms", 1000);

        // Head pose (6DRepNet)
        head_pose_enable_       = declare_parameter<bool>("head_pose_enable", false);
        head_pose_engine_path_  = declare_parameter<std::string>("head_pose_engine_path", "");
        head_pose_min_face_px_  = declare_parameter<int>("head_pose_min_face_px", 36);
        head_pose_expand_ratio_ = static_cast<float>(declare_parameter<double>("head_pose_expand_ratio", 0.12));
        head_pose_expand_ratio_ = std::clamp(head_pose_expand_ratio_, 0.f, 0.5f);
        head_pose_skip_yaw_deg_ = static_cast<float>(declare_parameter<double>("head_pose_skip_yaw_deg", 85.0));

        // Engagement
        engagement_topic_       = declare_parameter<std::string>("engagement_topic", "/human_face_fusion/scene_perception");
        publish_engagement_     = declare_parameter<bool>("publish_engagement", true);
        yaw_engaged_deg_        = static_cast<float>(declare_parameter<double>("yaw_engaged_deg",   30.0));
        pitch_engaged_deg_      = static_cast<float>(declare_parameter<double>("pitch_engaged_deg", 25.0));
        yaw_attention_deg_      = static_cast<float>(declare_parameter<double>("yaw_attention_deg",   55.0));
        pitch_attention_deg_    = static_cast<float>(declare_parameter<double>("pitch_attention_deg", 40.0));
        yaw_engaged_deg_     = std::clamp(yaw_engaged_deg_,     5.f, 85.f);
        pitch_engaged_deg_   = std::clamp(pitch_engaged_deg_,   5.f, 85.f);
        yaw_attention_deg_   = std::clamp(yaw_attention_deg_,   yaw_engaged_deg_,   89.f);
        pitch_attention_deg_ = std::clamp(pitch_attention_deg_, pitch_engaged_deg_, 89.f);
        engaged_max_distance_m_   = static_cast<float>(declare_parameter<double>("engaged_max_distance_m",   2.5));
        attention_max_distance_m_ = static_cast<float>(declare_parameter<double>("attention_max_distance_m", 4.0));
        engaged_max_distance_m_   = std::clamp(engaged_max_distance_m_,   0.1f, 20.f);
        attention_max_distance_m_ = std::clamp(attention_max_distance_m_, engaged_max_distance_m_, 20.f);

        // Face recognition (ArcFace)
        face_recog_enable_       = declare_parameter<bool>("face_recog_enable", false);
        face_recog_engine_path_  = declare_parameter<std::string>("face_recog_engine_path", "");
        face_db_path_            = declare_parameter<std::string>("face_db_path", "");
        face_recog_threshold_    = static_cast<float>(declare_parameter<double>("face_recog_threshold", 0.45));
        face_recog_threshold_    = std::clamp(face_recog_threshold_, 0.05f, 0.99f);

        // ── NEW: Tracker parameters ───────────────────────────────────────────
        tracker_iou_threshold_  = static_cast<float>(declare_parameter<double>("tracker_iou_threshold", 0.30));
        tracker_max_age_frames_ = declare_parameter<int>("tracker_max_age_frames", 30);
        tracker_ = std::make_unique<IouTracker>(tracker_iou_threshold_,
                                                std::max(1, tracker_max_age_frames_));

        // ── NEW: Quality gate parameters ──────────────────────────────────────
        recog_min_face_px_      = declare_parameter<int>("recog_min_face_px", 64);
        recog_min_face_conf_    = static_cast<float>(declare_parameter<double>("recog_min_face_conf", 0.75));
        recog_max_yaw_deg_      = static_cast<float>(declare_parameter<double>("recog_max_yaw_deg", 30.0));
        recog_max_pitch_deg_    = static_cast<float>(declare_parameter<double>("recog_max_pitch_deg", 25.0));
        recog_min_track_frames_ = declare_parameter<int>("recog_min_track_frames", 20);
        recog_interval_frames_  = declare_parameter<int>("recog_interval_frames", 150);
        recog_emb_buffer_size_  = declare_parameter<int>("recog_emb_buffer_size", 5);
        recog_emb_buffer_size_  = std::clamp(recog_emb_buffer_size_, 1, 10);

        // ── NEW: Gender detection ─────────────────────────────────────────────
        gender_enable_       = declare_parameter<bool>("gender_enable", false);
        gender_engine_path_  = declare_parameter<std::string>("gender_engine_path", "");
        // 模型约定：output[0]=female_logit, output[1]=male_logit（InsightFace 标准）
        // 不需要翻转；如果换了不同模型判反了，可以设为 true
        gender_swap_labels_  = declare_parameter<bool>("gender_swap_labels", false);
        // 低于此置信度不显示性别徽标（预处理修正后正常人脸应能达到 0.85+）
        gender_min_conf_     = declare_parameter<double>("gender_min_conf", 0.70);
        // 性别已确定后，按固定间隔重判，避免首轮误判长期固化
        gender_recheck_interval_frames_ =
            declare_parameter<int>("gender_recheck_interval_frames", 90);
        gender_recheck_interval_frames_ = std::clamp(gender_recheck_interval_frames_, 10, 600);

        // ── Initialise SCRFD ──────────────────────────────────────────────────
        if (engine_path_.empty())
            throw std::runtime_error("engine_path empty (SCRFD .trt path required)");
        detector_ = std::make_unique<SCRFD_TRT>(engine_path_);
        {
            std::string pre = declare_parameter<std::string>("scrfd_preprocess", "insightface");
            if (toLower(pre) == "namdvt" || toLower(pre) == "namdvt_upstream" || toLower(pre) == "upstream") {
                detector_->setPreprocess(SCRFD_TRT::Preprocess::NamdvtUpstream);
                RCLCPP_INFO(get_logger(), "SCRFD 预处理=namdvt_upstream");
            } else {
                detector_->setPreprocess(SCRFD_TRT::Preprocess::InsightFacePython);
                RCLCPP_INFO(get_logger(), "SCRFD 预处理=insightface");
            }
        }

        // ── Initialise 6DRepNet ───────────────────────────────────────────────
        if (head_pose_enable_) {
            if (head_pose_engine_path_.empty()) {
                RCLCPP_WARN(get_logger(), "head_pose_enable=true 但 head_pose_engine_path 为空，已禁用");
                head_pose_enable_ = false;
            } else {
                try {
                    head_pose_estimator_ = std::make_unique<SixDRepNet_TRT>(head_pose_engine_path_);
                    RCLCPP_INFO(get_logger(), "6DRepNet 已加载：%s", head_pose_engine_path_.c_str());
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "6DRepNet 加载失败：%s", e.what());
                    head_pose_enable_ = false;
                }
            }
        }

        // ── Initialise ArcFace + FaceDatabase ────────────────────────────────
        if (face_recog_enable_) {
            if (face_recog_engine_path_.empty()) {
                RCLCPP_WARN(get_logger(), "face_recog_enable=true 但 face_recog_engine_path 为空，已禁用");
                face_recog_enable_ = false;
            } else {
                try {
                    arcface_ = std::make_unique<ArcFaceTRT>(face_recog_engine_path_, face_recog_threshold_);
                    face_db_ = std::make_unique<FaceDatabase>();
                    int n_persons = 0;
                    if (!face_db_path_.empty()) {
                        n_persons = face_db_.get()->open(face_db_path_);
                        if (n_persons < 0) {
                            RCLCPP_WARN(get_logger(), "人脸库打开失败（%s），以空库启动",
                                        face_db_path_.c_str());
                            n_persons = 0;
                        }
                    } else {
                        RCLCPP_WARN(get_logger(), "face_db_path 为空，ArcFace 已加载但无持久化库");
                    }
                    RCLCPP_INFO(get_logger(),
                        "ArcFace 已加载：%s  persons=%d  threshold=%.2f  "
                        "质量门控: face_conf>=%.2f |yaw|<%.0f° |pitch|<%.0f° "
                        "face_px>=%d track_frames>=%d emb_buf=%d recheck=%d帧",
                        face_recog_engine_path_.c_str(), n_persons,
                        static_cast<double>(face_recog_threshold_),
                        static_cast<double>(recog_min_face_conf_),
                        static_cast<double>(recog_max_yaw_deg_),
                        static_cast<double>(recog_max_pitch_deg_),
                        recog_min_face_px_, recog_min_track_frames_,
                        recog_emb_buffer_size_, recog_interval_frames_);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "ArcFace 加载失败：%s，识别已禁用", e.what());
                    face_recog_enable_ = false;
                }
            }
        }

        // ── Initialise GenderAge ──────────────────────────────────────────────
        if (gender_enable_) {
            if (gender_engine_path_.empty()) {
                RCLCPP_WARN(get_logger(), "gender_enable=true 但 gender_engine_path 为空，已禁用");
                gender_enable_ = false;
            } else {
                try {
                    gender_model_ = std::make_unique<GenderAgeTRT>(gender_engine_path_);
                    RCLCPP_INFO(get_logger(), "GenderAge 已加载：%s  input=%dx%d",
                                gender_engine_path_.c_str(),
                                gender_model_->inputWidth(), gender_model_->inputHeight());
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "GenderAge 加载失败：%s，性别检测已禁用", e.what());
                    gender_enable_ = false;
                }
            }
        }

        // ── Publishers ────────────────────────────────────────────────────────
        rclcpp::QoS pub_qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default), rmw_qos_profile_default);
        pub_qos.keep_last(4);
        if (publish_fused_)
            pub_ = create_publisher<sensor_msgs::msg::Image>(fused_topic_, pub_qos);
        if (publish_engagement_) {
            rclcpp::QoS eng_qos(10); eng_qos.reliable();
            engagement_pub_ = create_publisher<remote_infer_bridge_cpp::msg::ScenePerceptionResult>(
                engagement_topic_, eng_qos);
        }

        // ── Subscriptions ─────────────────────────────────────────────────────
        rclcpp::QoS det_qos(10); det_qos.reliable();
        det_sub_ = create_subscription<remote_infer_bridge_cpp::msg::ScenePerceptionResult>(
            detections_topic_, det_qos,
            std::bind(&HumanFaceFusionNode::onDetections, this, std::placeholders::_1));
        color_qos_ = makeColorSubQoS(color_sub_qos_mode_);
        addColorSub(image_topic_);
        static constexpr const char* kOrbbec = "/camera/camera/color/image_raw";
        if (also_subscribe_orbbec_composable_color_ && image_topic_ != kOrbbec)
            addColorSub(kOrbbec);

        // ── UpdatePersonName service ──────────────────────────────────────────
        update_name_srv_ = create_service<remote_infer_bridge_cpp::srv::UpdatePersonName>(
            "/human_face_fusion/update_person_name",
            std::bind(&HumanFaceFusionNode::handleUpdatePersonName, this,
                      std::placeholders::_1, std::placeholders::_2));
        RCLCPP_INFO(get_logger(), "UpdatePersonName 服务：/human_face_fusion/update_person_name");

        RCLCPP_INFO(get_logger(),
            "融合节点启动 | 彩色=%s 检测=%s 追踪=IoU(iou=%.2f age=%d) "
            "识别=%s 性别=%s",
            image_topic_.c_str(), detections_topic_.c_str(),
            static_cast<double>(tracker_iou_threshold_), tracker_max_age_frames_,
            face_recog_enable_ ? "开" : "关",
            gender_enable_     ? "开" : "关");
    }

private:
    // ── Engagement ────────────────────────────────────────────────────────────
    bool usePersonHeadRoiInfer(bool use_dets, const remote_infer_bridge_cpp::msg::ScenePerceptionResult& fr) const {
        return use_dets && !fr.persons.empty();
    }

    uint8_t computeEngagement(float yaw, float pitch, float distance) const {
        using Msg = remote_infer_bridge_cpp::msg::ScenePerceptionResult;
        if (distance <= 0.f || distance > attention_max_distance_m_) return Msg::NOT_ENGAGED;
        const bool in_engaged  = std::abs(yaw) <= yaw_engaged_deg_  && std::abs(pitch) <= pitch_engaged_deg_;
        const bool in_attention= std::abs(yaw) <= yaw_attention_deg_ && std::abs(pitch) <= pitch_attention_deg_;
        if (in_engaged && distance <= engaged_max_distance_m_)  return Msg::ENGAGED;
        if (in_attention) return Msg::ATTENTION;
        return Msg::NOT_ENGAGED;
    }

    // ── Quality gate for ArcFace / registration ───────────────────────────────
    bool passesQualityGate(float face_conf, float yaw, float pitch,
                           int face_w, int face_h, int track_total_frames) const {
        if (yaw   > 900.f || pitch > 900.f) return false;  // pose not available yet
        return face_conf   >= recog_min_face_conf_  &&
               std::abs(yaw)   < recog_max_yaw_deg_   &&
               std::abs(pitch) < recog_max_pitch_deg_  &&
               face_w     >= recog_min_face_px_       &&
               face_h     >= recog_min_face_px_       &&
               track_total_frames >= recog_min_track_frames_;
    }

    // ── Recognition state machine update ──────────────────────────────────────
    void processRecognition(const cv::Mat& aligned, float face_conf,
                             float yaw, float pitch,
                             FaceTrack& track, int frame_number) {
        if (!arcface_ || !face_db_) return;
        if (aligned.empty()) return;

        switch (track.recog_state) {

        case TrackRecogState::PENDING: {
            // Collect embeddings while quality gate passes
            FaceEmbedding emb;
            if (!arcface_->extractEmbedding(aligned, emb)) return;

            std::array<float, FaceTrack::kEmbDim> arr;
            std::memcpy(arr.data(), emb.v, FaceTrack::kEmbDim * sizeof(float));
            track.reg_emb_buffer.push_back(arr);
            track.reg_yaw_buffer.push_back(yaw);
            track.reg_pitch_buffer.push_back(pitch);
            track.reg_conf_buffer.push_back(face_conf);

            if (static_cast<int>(track.reg_emb_buffer.size()) < recog_emb_buffer_size_)
                return;  // keep accumulating

            // Buffer full → run identify with the last (freshest) embedding
            const FaceMatch match = face_db_->identify(emb, face_recog_threshold_);
            track.last_recog_frame = frame_number;

            if (match.identified) {
                track.person_uuid  = match.uuid;
                track.person_name  = match.name;
                track.recog_conf   = match.similarity;
                track.recog_state  = TrackRecogState::IDENTIFIED;
                face_db_->touchPerson(match.uuid);
                RCLCPP_INFO(get_logger(),
                    "[Track %d] 识别成功: uuid=%.8s name='%s' conf=%.1f%%",
                    track.track_id, match.uuid.c_str(), match.name.c_str(),
                    static_cast<double>(match.similarity) * 100.0);
            } else {
                // Not in DB → auto-register
                std::vector<FaceEmbedding> embs;
                embs.reserve(track.reg_emb_buffer.size());
                for (const auto& a : track.reg_emb_buffer) {
                    FaceEmbedding e;
                    std::memcpy(e.v, a.data(), FaceTrack::kEmbDim * sizeof(float));
                    embs.push_back(e);
                }
                const std::string uuid = face_db_->registerPerson(
                    "", embs, track.reg_yaw_buffer,
                    track.reg_pitch_buffer, track.reg_conf_buffer);
                if (!uuid.empty()) {
                    track.person_uuid  = uuid;
                    track.person_name  = "";
                    track.recog_conf   = 0.f;
                    track.recog_state  = TrackRecogState::IDENTIFIED;
                    RCLCPP_INFO(get_logger(),
                        "[Track %d] 新人注册: uuid=%.8s (name 待状态机填写)",
                        track.track_id, uuid.c_str());
                }
            }
            // Clear buffer regardless of outcome
            track.reg_emb_buffer.clear();
            track.reg_yaw_buffer.clear();
            track.reg_pitch_buffer.clear();
            track.reg_conf_buffer.clear();
            break;
        }

        case TrackRecogState::IDENTIFIED: {
            // Periodic re-verification
            if ((frame_number - track.last_recog_frame) < recog_interval_frames_) return;

            FaceEmbedding emb;
            if (!arcface_->extractEmbedding(aligned, emb)) return;
            const FaceMatch match = face_db_->identify(emb, face_recog_threshold_);
            track.last_recog_frame = frame_number;

            if (match.identified) {
                if (match.uuid != track.person_uuid) {
                    RCLCPP_INFO(get_logger(),
                        "[Track %d] 身份变更: %.8s → %.8s",
                        track.track_id, track.person_uuid.c_str(), match.uuid.c_str());
                    track.person_uuid = match.uuid;
                    track.person_name = match.name;
                }
                track.recog_conf = match.similarity;
                face_db_->touchPerson(match.uuid);
            } else {
                // Re-verification failed → reset to PENDING
                RCLCPP_INFO(get_logger(),
                    "[Track %d] 重验失败(sim=%.2f)，重置为PENDING",
                    track.track_id, static_cast<double>(match.similarity));
                track.recog_state  = TrackRecogState::PENDING;
                track.person_uuid  = "";
                track.person_name  = "";
                track.recog_conf   = 0.f;
            }
            break;
        }
        }
    }

    // ── Core enrichment: SCRFD + 6DRepNet + ArcFace + GenderAge ──────────────
    void enrichPersonsWithFaceAndPose(
        const cv::Mat& img,
        remote_infer_bridge_cpp::msg::ScenePerceptionResult& fr,
        const std::vector<FaceTrack*>& tracks)
    {
        const int W = img.cols, H = img.rows;
        const int n_persons = static_cast<int>(fr.persons.size());
        int n_rois = 0;

        for (int pi = 0; pi < n_persons && n_rois < face_max_person_rois_; ++pi) {
            auto& p = fr.persons[static_cast<size_t>(pi)];
            FaceTrack* track = (pi < static_cast<int>(tracks.size())) ? tracks[static_cast<size_t>(pi)] : nullptr;

            // ── Assign track_id ───────────────────────────────────────────────
            if (track) p.track_id = track->track_id;

            // ── SCRFD: face detection in head-shoulder ROI ────────────────────
            const cv::Rect roi = personHeadCropRect(p, W, H,
                person_head_height_ratio_, person_head_width_pad_ratio_,
                person_head_top_expand_ratio_);
            ++n_rois;
            if (roi.width < face_roi_min_side_ || roi.height < face_roi_min_side_) continue;

            std::vector<FaceObject> loc;
            detector_->detect(img(roi), loc, roi_scrfd_prob_threshold_, roi_scrfd_nms_threshold_);
            if (loc.empty()) continue;

            const auto& best = *std::max_element(loc.begin(), loc.end(),
                [](const FaceObject& a, const FaceObject& b){ return a.prob < b.prob; });
            FaceObject gf = best;
            offsetFaceToGlobal(gf, static_cast<float>(roi.x), static_cast<float>(roi.y));

            p.has_face = true;
            p.face_x   = static_cast<int32_t>(gf.rect.x);
            p.face_y   = static_cast<int32_t>(gf.rect.y);
            p.face_w   = static_cast<int32_t>(gf.rect.width);
            p.face_h   = static_cast<int32_t>(gf.rect.height);
            p.face_conf= gf.prob;

            // ── Head pose (6DRepNet) ──────────────────────────────────────────
            if (head_pose_enable_ && head_pose_estimator_) {
                const cv::Rect expanded = expandFaceRect(gf.rect, head_pose_expand_ratio_, W, H);
                if (expanded.width >= head_pose_min_face_px_ && expanded.height >= head_pose_min_face_px_) {
                    try {
                        const HeadPose hp = head_pose_estimator_->predict(img(expanded));
                        p.yaw   = hp.yaw;
                        p.pitch = hp.pitch;
                        p.roll  = hp.roll;
                        p.engagement = computeEngagement(p.yaw, p.pitch, p.distance);
                    } catch (const std::exception& e) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                             "6DRepNet predict: %s", e.what());
                        p.yaw = p.pitch = 999.f;
                        p.engagement = remote_infer_bridge_cpp::msg::ScenePerceptionResult::NOT_ENGAGED;
                    }
                } else {
                    p.yaw = p.pitch = 999.f;
                    p.engagement = remote_infer_bridge_cpp::msg::ScenePerceptionResult::NOT_ENGAGED;
                }
            } else {
                p.yaw = 999.f; p.pitch = 999.f; p.roll = 0.f;
                p.engagement = remote_infer_bridge_cpp::msg::ScenePerceptionResult::NOT_ENGAGED;
            }

            // ── Aligned face (shared by ArcFace + Gender) ────────────────────
            cv::Mat aligned;
            if ((face_recog_enable_ && arcface_) || (gender_enable_ && gender_model_)) {
                aligned = ArcFaceTRT::alignFace(img, gf.landmark);
            }

            // ── Gender detection：质量门控 + 多帧投票 ────────────────────────
            // 使用与 ArcFace 相同的关键点对齐图（姿态归一化），确保每次输入一致。
            // 累积 kGenderVotesNeeded 票后取多数，避免单帧误判。
            if (gender_enable_ && gender_model_ && track && !aligned.empty()) {
                const bool quality_ok = passesQualityGate(
                    gf.prob, p.yaw, p.pitch, p.face_w, p.face_h, track->total_frames);
                const bool need_initial_vote = !track->gender_done;
                const bool need_periodic_recheck =
                    track->gender_done &&
                    track->gender_last_eval_frame >= 0 &&
                    (frame_number_ - track->gender_last_eval_frame) >= gender_recheck_interval_frames_;
                if (quality_ok && (need_initial_vote || need_periodic_recheck)) {
                    // 首次存图：把模型实际看到的对齐脸存到 /tmp 方便肉眼排查
                    {
                        static std::set<int> saved_tracks;
                        if (saved_tracks.find(track->track_id) == saved_tracks.end()) {
                            saved_tracks.insert(track->track_id);
                            std::string path = "/tmp/gender_face_track" +
                                               std::to_string(track->track_id) + ".png";
                            cv::imwrite(path, aligned);
                            RCLCPP_INFO(get_logger(),
                                "[GenderDBG] track=%d 对齐脸已存 %s  yaw=%.1f pitch=%.1f",
                                track->track_id, path.c_str(), p.yaw, p.pitch);
                        }
                    }
                    GenderResult gr = gender_model_->predict(aligned);
                    if (gr.gender == GenderResult::UNKNOWN) {
                        // 模型本帧未给出有效性别，不更新状态
                    } else if (need_initial_vote) {
                        track->gender_last_eval_frame = frame_number_;
                        if (gr.gender == GenderResult::MALE)   track->gender_votes_male++;
                        if (gr.gender == GenderResult::FEMALE) track->gender_votes_female++;

                        const int total = track->gender_votes_male + track->gender_votes_female;
                        if (total >= FaceTrack::kGenderVotesNeeded) {
                            uint8_t majority = (track->gender_votes_male >= track->gender_votes_female)
                                               ? GenderResult::MALE : GenderResult::FEMALE;
                            if (gender_swap_labels_) {
                                majority = (majority == GenderResult::MALE)
                                           ? GenderResult::FEMALE : GenderResult::MALE;
                            }
                            const float conf = static_cast<float>(
                                std::max(track->gender_votes_male, track->gender_votes_female)) / total;

                            // 置信度不足时不显示徽标（UNKNOWN），避免弱预测误导
                            track->gender      = (conf >= gender_min_conf_) ? majority
                                                                             : GenderResult::UNKNOWN;
                            track->gender_conf = conf;
                            track->gender_done = true;
                            RCLCPP_INFO(get_logger(),
                                "[Track %d] 性别%s: %s (M%d/F%d  conf=%.0f%%)",
                                track->track_id,
                                conf >= gender_min_conf_ ? "确定" : "不确定(跳过)",
                                majority == GenderResult::MALE ? "男" : "女",
                                track->gender_votes_male, track->gender_votes_female,
                                conf * 100.f);
                        }
                    } else if (need_periodic_recheck) {
                        track->gender_last_eval_frame = frame_number_;
                        uint8_t pred = gr.gender;
                        if (gender_swap_labels_) {
                            pred = (pred == GenderResult::MALE)
                                 ? GenderResult::FEMALE : GenderResult::MALE;
                        }
                        if (gr.conf >= gender_min_conf_) {
                            track->gender = pred;
                            track->gender_conf = gr.conf;
                            RCLCPP_DEBUG(get_logger(),
                                "[Track %d] 性别重判更新: %s conf=%.0f%%",
                                track->track_id,
                                pred == GenderResult::MALE ? "男" : "女",
                                gr.conf * 100.f);
                        }
                    }
                }
            }

            // ── ArcFace recognition (state-machine driven) ───────────────────
            if (face_recog_enable_ && arcface_ && track && !aligned.empty()) {
                const bool should_run = [&]() -> bool {
                    switch (track->recog_state) {
                    case TrackRecogState::PENDING:
                        return passesQualityGate(gf.prob, p.yaw, p.pitch,
                                                 p.face_w, p.face_h, track->total_frames);
                    case TrackRecogState::IDENTIFIED:
                        return (frame_number_ - track->last_recog_frame) >= recog_interval_frames_ &&
                               passesQualityGate(gf.prob, p.yaw, p.pitch,
                                                 p.face_w, p.face_h, track->total_frames);
                    }
                    return false;
                }();

                if (should_run) {
                    processRecognition(aligned, gf.prob, p.yaw, p.pitch, *track, frame_number_);
                }

                // Also fill face_embedding in msg regardless of recog state (if aligned ok)
                if (track->recog_state == TrackRecogState::IDENTIFIED || should_run) {
                    FaceEmbedding emb;
                    if (arcface_->extractEmbedding(aligned, emb)) {
                        static_assert(FaceEmbedding::DIM == 512, "msg face_embedding is [512]");
                        for (int d = 0; d < FaceEmbedding::DIM; ++d)
                            p.face_embedding[d] = emb.v[d];
                    }
                }
            }

            // ── Populate message fields from track ───────────────────────────
            if (track) {
                p.track_id        = track->track_id;
                p.person_uuid     = track->person_uuid;
                p.person_name     = track->person_name;
                p.face_recog_conf = track->recog_conf;
                p.gender          = track->gender;
                p.gender_conf     = track->gender_conf;
            }
        }
    }

    void addColorSub(const std::string& topic) {
        if (topic.empty() || color_topics_.count(topic)) return;
        color_topics_.insert(topic);
        color_subs_.push_back(create_subscription<sensor_msgs::msg::Image>(
            topic, color_qos_,
            std::bind(&HumanFaceFusionNode::onImage, this, std::placeholders::_1)));
        RCLCPP_INFO(get_logger(), "订阅彩色 %s", topic.c_str());
    }

    void onDetections(const remote_infer_bridge_cpp::msg::ScenePerceptionResult::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_dets_ = *msg;
        last_dets_time_ = std::chrono::steady_clock::now();
        if (!msg->persons.empty()) {
            last_nonempty_persons_msg_ = *msg;
            last_nonempty_persons_time_ = last_dets_time_;
        }
    }

    // ── UpdatePersonName service handler ──────────────────────────────────────
    void handleUpdatePersonName(
        const std::shared_ptr<remote_infer_bridge_cpp::srv::UpdatePersonName::Request> req,
        std::shared_ptr<remote_infer_bridge_cpp::srv::UpdatePersonName::Response> res)
    {
        if (!face_db_ || !face_db_->isOpen()) {
            res->success = false;
            res->message = "face_db not open";
            return;
        }
        if (req->person_uuid.empty()) {
            res->success = false;
            res->message = "person_uuid is empty";
            return;
        }
        const bool ok = face_db_->updateName(req->person_uuid, req->name);
        res->success = ok;
        res->message = ok ? "ok" : "uuid not found";
        if (ok) {
            RCLCPP_INFO(get_logger(), "UpdatePersonName: uuid=%.8s  name='%s'",
                        req->person_uuid.c_str(), req->name.c_str());
            // Sync in-memory tracks immediately so the display updates without
            // waiting for the next re-verification cycle.
            if (tracker_) {
                tracker_->updatePersonNameByUuid(req->person_uuid, req->name);
            }
        }
    }

    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat img;
        if (!decodeColorImage(msg, img)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                                 "图像解码失败 encoding=%s", msg->encoding.c_str());
            return;
        }
        ensureBgrU8C3(img);
        if (img.empty()) return;
        const auto t1 = std::chrono::steady_clock::now();

        remote_infer_bridge_cpp::msg::ScenePerceptionResult fr_copy;
        remote_infer_bridge_cpp::msg::ScenePerceptionResult nonempty_persons_snapshot;
        std::chrono::steady_clock::time_point nonempty_persons_snapshot_time{};
        bool use_dets = false;
        double body_ms_for_pipeline = 0.;
        bool body_timing_stamp_align = false;
        bool body_timing_legacy = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_dets_time_).count();
            const bool stale_ok = (detection_stale_ms_ < 0) || (age_ms <= detection_stale_ms_);
            const bool det_meta_ok = last_dets_.image_width > 0u && last_dets_.image_height > 0u &&
                                     img.cols > 0 && img.rows > 0;
            const bool dims_exact = det_meta_ok &&
                last_dets_.image_width == static_cast<uint32_t>(img.cols) &&
                last_dets_.image_height == static_cast<uint32_t>(img.rows);
            const bool dims_ok = dims_exact || (align_detection_dims_ && det_meta_ok);
            if (stale_ok && det_meta_ok && last_dets_.body_pipeline_ms > 0.f) {
                body_ms_for_pipeline = static_cast<double>(last_dets_.body_pipeline_ms);
                body_timing_stamp_align =
                    (last_dets_.header.stamp.sec == msg->header.stamp.sec &&
                     last_dets_.header.stamp.nanosec == msg->header.stamp.nanosec);
                body_timing_legacy = last_dets_.header.stamp.sec == 0 &&
                                     last_dets_.header.stamp.nanosec == 0;
            }
            if (stale_ok && dims_ok && !last_dets_.persons.empty()) {
                fr_copy = last_dets_;
                if (!dims_exact && align_detection_dims_) {
                    scalePersonsToImage(fr_copy, img.cols, img.rows);
                }
                use_dets = true;
            }
            nonempty_persons_snapshot = last_nonempty_persons_msg_;
            nonempty_persons_snapshot_time = last_nonempty_persons_time_;
        }

        remote_infer_bridge_cpp::msg::ScenePerceptionResult fr_face = fr_copy;
        bool use_roi_for_face = usePersonHeadRoiInfer(use_dets, fr_face);
        if (!use_roi_for_face && face_persons_fallback_ms_ > 0) {
            const auto fb_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - nonempty_persons_snapshot_time).count();
            if (!nonempty_persons_snapshot.persons.empty() && fb_age_ms >= 0 &&
                fb_age_ms <= face_persons_fallback_ms_) {
                fr_face = nonempty_persons_snapshot;
                if (align_detection_dims_ && fr_face.image_width > 0u &&
                    (fr_face.image_width != static_cast<uint32_t>(img.cols) ||
                     fr_face.image_height != static_cast<uint32_t>(img.rows)))
                    scalePersonsToImage(fr_face, img.cols, img.rows);
                use_roi_for_face = usePersonHeadRoiInfer(true, fr_face);
            }
        }

        // ── IoU tracking ──────────────────────────────────────────────────────
        std::vector<FaceTrack*> tracks_for_person;
        if (use_roi_for_face && tracker_) {
            std::vector<cv::Rect> bboxes;
            bboxes.reserve(fr_face.persons.size());
            for (const auto& p : fr_face.persons)
                bboxes.push_back({p.body_x, p.body_y, p.body_w, p.body_h});
            tracks_for_person = tracker_->update(bboxes, frame_number_);
        }

        // ── Face / pose / recog / gender enrichment ───────────────────────────
        const auto t_face0 = std::chrono::steady_clock::now();
        try {
            if (use_roi_for_face)
                enrichPersonsWithFaceAndPose(img, fr_face, tracks_for_person);
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                                  "SCRFD/6DRepNet/ArcFace: %s", e.what());
        }
        const auto t2 = std::chrono::steady_clock::now();

        // ── Visualisation ──────────────────────────────────────────────────────
        cv::Mat viz = img.clone();
        std::vector<cv::Rect> occ;
        buildOccupiedFromDetections(viz, use_dets ? &fr_copy : nullptr, use_dets, occ);
        if (use_dets) drawPersonBoxes(viz, use_roi_for_face ? fr_face : fr_copy);
        drawFaceBoxes(viz, fr_face.persons);

        if (head_pose_enable_ && head_pose_estimator_) {
            for (const auto& p : fr_face.persons) {
                if (!p.has_face) continue;
                const cv::Rect face_rect(p.face_x, p.face_y, p.face_w, p.face_h);
                const cv::Rect expanded = expandFaceRect(face_rect, head_pose_expand_ratio_, viz.cols, viz.rows);
                if (expanded.width >= head_pose_min_face_px_ && expanded.height >= head_pose_min_face_px_) {
                    const HeadPose hp{p.yaw, p.pitch, p.roll};
                    head_pose_estimator_->drawAxis(viz, face_rect, hp, head_pose_skip_yaw_deg_);
                }
            }
        }

        // ── Publish ScenePerceptionResult ──────────────────────────────────────
        if (engagement_pub_) {
            using Msg = remote_infer_bridge_cpp::msg::ScenePerceptionResult;
            Msg emsg = fr_face;
            emsg.header = msg->header;
            emsg.image_width  = static_cast<uint32_t>(img.cols);
            emsg.image_height = static_cast<uint32_t>(img.rows);
            uint8_t best = Msg::NOT_ENGAGED;
            uint32_t n_engaged = 0, n_attention = 0;
            float closest_engaged = -1.f, closest_attention = -1.f;
            for (const auto& p : emsg.persons) {
                if (!p.has_face) continue;
                if (p.engagement > best) best = p.engagement;
                if (p.engagement == Msg::ENGAGED) {
                    ++n_engaged;
                    if (p.distance > 0.f && (closest_engaged < 0.f || p.distance < closest_engaged))
                        closest_engaged = p.distance;
                } else if (p.engagement == Msg::ATTENTION) {
                    ++n_attention;
                    if (p.distance > 0.f && (closest_attention < 0.f || p.distance < closest_attention))
                        closest_attention = p.distance;
                }
            }
            emsg.best_engagement            = best;
            emsg.engaged_count              = n_engaged;
            emsg.attention_count            = n_attention;
            emsg.closest_engaged_distance   = closest_engaged;
            emsg.closest_attention_distance = closest_attention;
            engagement_pub_->publish(emsg);
        }

        if (use_dets) drawPersonLabels(viz, use_roi_for_face ? fr_face : fr_copy, occ);
        drawFaceLabels(viz, fr_face.persons, occ);
        // 性别徽标：♂ 蓝色 / ♀ 粉色（覆盖在人脸框左上角）
        if (gender_enable_) {
            for (const auto& p : fr_face.persons) drawGenderBadge(viz, p);
        }
        if (viz_show_legend_) drawFusionLegendCompact(viz);

        // ── Publish fused image ────────────────────────────────────────────────
        double ros_stamp_to_pub_ms = -1.;
        std::chrono::steady_clock::time_point t_after_fused_pub{};
        bool fused_image_published = false;
        if (publish_fused_ && pub_) {
            if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
                try {
                    const rclcpp::Time t_stamp(msg->header.stamp);
                    ros_stamp_to_pub_ms = (this->now() - t_stamp).seconds() * 1000.0;
                } catch (...) {}
            }
            std_msgs::msg::Header hdr;
            hdr.stamp    = msg->header.stamp;
            hdr.frame_id = msg->header.frame_id;
            cv_bridge::CvImage cv_img(hdr, sensor_msgs::image_encodings::BGR8, viz);
            pub_->publish(*cv_img.toImageMsg());
            t_after_fused_pub = std::chrono::steady_clock::now();
            fused_image_published = true;
        }

        if (show_window_) { cv::imshow(window_name_, viz); cv::waitKey(1); }

        const auto t3 = std::chrono::steady_clock::now();
        const size_t n_faces = static_cast<size_t>(
            std::count_if(fr_face.persons.begin(), fr_face.persons.end(),
                          [](const auto& p){ return p.has_face; }));

        if (log_frame_timing_) {
            const double ms_decode    = std::chrono::duration<double,std::milli>(t1-t0).count();
            const double ms_prep      = std::chrono::duration<double,std::milli>(t_face0-t1).count();
            const double ms_face      = std::chrono::duration<double,std::milli>(t2-t_face0).count();
            const double ms_draw      = std::chrono::duration<double,std::milli>(t3-t2).count();
            const double ms_total     = std::chrono::duration<double,std::milli>(t3-t0).count();
            const double ms_to_pub    = fused_image_published ?
                std::chrono::duration<double,std::milli>(t_after_fused_pub-t0).count() : -1.;
            const double fusion_for_series = fused_image_published ? ms_to_pub : ms_total;
            char line[896];
            if (body_ms_for_pipeline > 0.) {
                const char* align_note = body_timing_stamp_align ? "同帧stamp"
                    : body_timing_legacy ? "stamp=0(legacy)" : "近似";
                std::snprintf(line, sizeof(line),
                    "[融合] 全帧=%.1fms(解码%.1f+准备%.1f+推理%.1f+叠图%.1f) "
                    "face=%zu track=%d | [串联] 人体%.1f+融合%.1f=%.1fms (%s)",
                    ms_total, ms_decode, ms_prep, ms_face, ms_draw, n_faces,
                    tracker_ ? tracker_->liveCount() : 0,
                    body_ms_for_pipeline, fusion_for_series,
                    body_ms_for_pipeline + fusion_for_series, align_note);
            } else {
                std::snprintf(line, sizeof(line),
                    "[融合] 全帧=%.1fms face=%zu track=%d | trt未提供body_pipeline_ms",
                    ms_total, n_faces, tracker_ ? tracker_->liveCount() : 0);
            }
            if (log_frame_timing_period_ms_ <= 0)
                RCLCPP_INFO(get_logger(), "%s", line);
            else
                RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), log_frame_timing_period_ms_, "%s", line);
        }

        ++frame_number_;
        ++frame_count_;
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last_log_).count();
        if (log_period_ms_ > 0 && dt >= static_cast<double>(log_period_ms_) / 1000.0) {
            RCLCPP_INFO(get_logger(), "脸=%zu 追踪=%d persons=%d ~%.1fHz",
                        n_faces,
                        tracker_ ? tracker_->liveCount() : 0,
                        static_cast<int>(fr_face.persons.size()),
                        static_cast<double>(frame_count_) / dt);
            frame_count_ = 0;
            last_log_ = now;
        }
    }

    // ── Members ───────────────────────────────────────────────────────────────
    std::unique_ptr<SCRFD_TRT>      detector_;
    std::unique_ptr<SixDRepNet_TRT> head_pose_estimator_;
    std::unique_ptr<ArcFaceTRT>     arcface_;
    std::unique_ptr<FaceDatabase>   face_db_;
    std::unique_ptr<GenderAgeTRT>   gender_model_;
    std::unique_ptr<IouTracker>     tracker_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::Publisher<remote_infer_bridge_cpp::msg::ScenePerceptionResult>::SharedPtr engagement_pub_;
    rclcpp::Subscription<remote_infer_bridge_cpp::msg::ScenePerceptionResult>::SharedPtr det_sub_;
    rclcpp::Service<remote_infer_bridge_cpp::srv::UpdatePersonName>::SharedPtr update_name_srv_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> color_subs_;
    std::set<std::string> color_topics_;
    rclcpp::QoS color_qos_{rclcpp::KeepLast(1)};

    std::mutex mutex_;
    remote_infer_bridge_cpp::msg::ScenePerceptionResult last_dets_;
    std::chrono::steady_clock::time_point last_dets_time_{};
    remote_infer_bridge_cpp::msg::ScenePerceptionResult last_nonempty_persons_msg_;
    std::chrono::steady_clock::time_point last_nonempty_persons_time_{};

    // ── Parameter cache ───────────────────────────────────────────────────────
    std::string engine_path_;
    std::string image_topic_;
    std::string detections_topic_;
    std::string fused_topic_;
    std::string window_name_;
    std::string color_sub_qos_mode_;
    float nms_threshold_{0.4f};
    int   detection_stale_ms_{500};
    double person_head_height_ratio_{0.58};
    double person_head_width_pad_ratio_{0.24};
    double person_head_top_expand_ratio_{0.14};
    int  face_roi_min_side_{48};
    int  face_max_person_rois_{8};
    bool align_detection_dims_{true};
    int  face_persons_fallback_ms_{350};
    float roi_scrfd_prob_threshold_{0.38f};
    float roi_scrfd_nms_threshold_{0.45f};
    int  log_period_ms_{2000};
    bool publish_fused_{true};
    bool show_window_{true};
    bool viz_show_legend_{false};
    bool also_subscribe_orbbec_composable_color_{false};
    bool log_frame_timing_{false};
    int  log_frame_timing_period_ms_{1000};

    // Head pose
    bool        head_pose_enable_{false};
    std::string head_pose_engine_path_;
    int         head_pose_min_face_px_{36};
    float       head_pose_expand_ratio_{0.12f};
    float       head_pose_skip_yaw_deg_{85.f};

    // Engagement
    std::string engagement_topic_;
    bool  publish_engagement_{true};
    float yaw_engaged_deg_{30.f};
    float pitch_engaged_deg_{25.f};
    float yaw_attention_deg_{55.f};
    float pitch_attention_deg_{40.f};
    float engaged_max_distance_m_{2.5f};
    float attention_max_distance_m_{4.0f};

    // Face recognition
    bool        face_recog_enable_{false};
    std::string face_recog_engine_path_;
    std::string face_db_path_;
    float       face_recog_threshold_{0.45f};

    // Tracker
    float tracker_iou_threshold_{0.30f};
    int   tracker_max_age_frames_{30};

    // Quality gate
    int   recog_min_face_px_{64};
    float recog_min_face_conf_{0.75f};
    float recog_max_yaw_deg_{30.f};
    float recog_max_pitch_deg_{25.f};
    int   recog_min_track_frames_{20};
    int   recog_interval_frames_{150};
    int   recog_emb_buffer_size_{5};

    // Gender
    bool        gender_enable_{false};
    bool        gender_swap_labels_{false};
    float       gender_min_conf_{0.55f};
    int         gender_recheck_interval_frames_{90};
    std::string gender_engine_path_;

    // Counters
    int    frame_number_{0};
    size_t frame_count_{0};
    std::chrono::steady_clock::time_point last_log_{std::chrono::steady_clock::now()};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<HumanFaceFusionNode>());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "human_face_fusion_node: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}












