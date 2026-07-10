import numpy as np
import torch

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)

x1 = torch.arange(6)

# 1

# 1a
x1 = x1.reshape(2,3)
print(x1)

# 1b
x2 = torch.zeros((4,5))
print(x2)

# 1c
x3 = torch.rand((3,3))
print(x3)


# 2

# 2a 
#[[1,1,1],[1,1,1]]

#2b
#[[1,1,1,1],[1,1,1,1],[1,1,1,1]]


# 3
board = torch.tensor([
    [1, 0, 2],
    [0, 2, 1],
    [1, 0, 0]
])

# 3.1
print(board[1,1])

# 3.2
print(board[-1])

# 3.3
print(board[:,0])

# 3.4
board[1,1] = 5


# 4
x = torch.arange(1, 13)
x = x.reshape(4,3)
# shape is (4,3)


# 5

# A * A
# [[1., 4.],[9., 16.]]

# A @ A
# [[7.,10.],[15., 22.]]

A = torch.tensor([[1,2],[3,4]])
B = torch.tensor([[5,6],[7,8]])

# 5a
print(torch.concatenate([A,B], dim= 0))

# 5b
print(torch.concatenate([A,B], dim= 1))


# 7
x7 = torch.randint(1,7,(100,))
values, counts = torch.unique(x7, return_counts=True)

print(x7)


print(f"Average: {x7.sum()/100}")
print(f"Number of sixes: {counts[5]}")

# Mini
tic_tac_toe = torch.tensor(
    [
        [1,0,2],
        [0,1,0],
        [2,0,0]
    ]
)

values, counts = torch.unique(tic_tac_toe, return_counts=True)

# Mini.1
print(f"X: {counts[1]}")

# Mini.2
print(f"O: {counts[2]}")

# Mini.3
if tic_tac_toe[1,1] == 0:
    print("Empty")
else:
    print("Not empty")

# Mini.4
tic_tac_toe[2,2] = 1

# Mini.5
print(tic_tac_toe)

# Bonus
def place_piece(board, row, col, player):
    if board[row,col] != 0:
        board[row,col] = player
    else:
        print("Not empty")

