//! `--dump`: render frames off-screen (for checking layout and making screenshots without a TTY).

use ratatui::{buffer::Buffer, style::Color};
use unicode_width::UnicodeWidthStr;

fn css(c: Color, default: &str) -> String {
    match c {
        Color::Reset => default.into(),
        Color::Black => "#1d1f21".into(),
        Color::Red => "#e5534b".into(),
        Color::Green => "#57ab5a".into(),
        Color::Yellow => "#d4a72c".into(),
        Color::Blue => "#539bf5".into(),
        Color::Magenta => "#b083f0".into(),
        Color::Cyan => "#39c5cf".into(),
        Color::Gray => "#adbac7".into(),
        Color::DarkGray => "#636e7b".into(),
        Color::White => "#ffffff".into(),
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        Color::Indexed(i @ 16..=231) => {
            let lv = |n: u8| if n == 0 { 0 } else { 55 + 40 * n as u32 };
            let i = i - 16;
            format!("#{:02x}{:02x}{:02x}", lv(i / 36), lv(i / 6 % 6), lv(i % 6))
        }
        Color::Indexed(i @ 232..=255) => {
            let v = 8 + 10 * (i as u32 - 232);
            format!("#{v:02x}{v:02x}{v:02x}")
        }
        _ => default.into(),
    }
}

pub fn text(buf: &Buffer) -> String {
    let mut out = String::new();
    for y in 0..buf.area.height {
        let mut line = String::new();
        let mut x = 0;
        while x < buf.area.width {
            let sym = buf[(x, y)].symbol();
            line.push_str(sym);
            // a wide (CJK) glyph covers the next cell too
            x += sym.width().max(1) as u16;
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

pub fn html(buf: &Buffer) -> String {
    let mut out = String::from(
        "<!doctype html><meta charset=utf-8><style>body{margin:0;background:#22272e}\
         pre{margin:0;padding:12px;font:13px/1.3 Menlo,monospace;color:#adbac7}</style><pre>",
    );
    for y in 0..buf.area.height {
        let mut x = 0;
        while x < buf.area.width {
            let c = &buf[(x, y)];
            x += c.symbol().width().max(1) as u16;
            let sym = match c.symbol() {
                "<" => "&lt;",
                ">" => "&gt;",
                "&" => "&amp;",
                s => s,
            };
            let bold = c.modifier.contains(ratatui::style::Modifier::BOLD);
            out.push_str(&format!(
                "<span style=\"color:{};background:{}{}\">{}</span>",
                css(c.fg, "#adbac7"),
                css(c.bg, "transparent"),
                if bold { ";font-weight:700" } else { "" },
                sym
            ));
        }
        out.push('\n');
    }
    out.push_str("</pre>");
    out
}
