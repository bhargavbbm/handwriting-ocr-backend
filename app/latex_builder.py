def build_latex(blocks):
    """
    Convert OCR blocks into a LaTeX document.
    Blocks: list of {type: 'text'/'equation', content: '...'}
    """

    output = []

    for block in blocks:
        if block["type"] == "text":
            output.append(block["content"] + "\n")

        elif block["type"] == "equation":
            output.append(f"$$ {block['content']} $$\n")

    return "".join(output)
