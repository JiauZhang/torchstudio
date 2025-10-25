import pymupdf

def draw_text(text, font_size=12, margin=50, line_spacing=1.5, font_name='helv'):
    doc = pymupdf.open()
    # A4 page
    page = doc.new_page(width=595, height=842)
    font = pymupdf.Font(fontname=font_name)
    page_width = page.rect.width
    available_width = page_width - 2 * margin
    x, y = margin, margin
    line_height = font_size * line_spacing
    original_lines = text.split('\n')

    for line in original_lines:
        # handle consecutive newline characters
        if not line.strip():
            y += line_height
            continue

        words = line.split()
        join_char = ' '

        current_line = []
        for word in words:
            test_line = join_char.join(current_line + [word])
            text_width = font.text_length(test_line, fontsize=font_size)
            
            if text_width <= available_width:
                current_line.append(word)
            else:
                line_text = join_char.join(current_line)
                page.insert_text(
                    (x, y),
                    line_text,
                    fontsize=font_size,
                    fontname=font_name,
                    color=(0, 0, 0)
                )
                y += line_height
                current_line = [word]

        if current_line:
            line_text = join_char.join(current_line)
            page.insert_text(
                (x, y),
                line_text,
                fontsize=font_size,
                fontname=font_name,
                color=(0, 0, 0)
            )
            y += line_height

    pil_image = page.get_pixmap().pil_image()
    doc.close()
    return pil_image
