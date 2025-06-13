def split_board_into_squares(board_img, grid_size=8):
    h, w = board_img.shape[:2]
    square_h = h // grid_size
    square_w = w // grid_size

    squares = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * square_w
            y1 = row * square_h
            x2 = x1 + square_w
            y2 = y1 + square_h
            square_img = board_img[y1:y2, x1:x2]
            squares.append({
                "row": row,
                "col": col,
                "image": square_img
            })
    return squares
