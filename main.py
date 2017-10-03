from board import SudokuBoard


def main():
    board = SudokuBoard('./images/tests/0.png')
    for column in board.get_board():
        print(column)

if __name__ == "__main__":
    main()
