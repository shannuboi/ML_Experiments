from numpy import random


moves = ["R", "L", "U", "D", "F", "B"]
idx_to_move = dict(enumerate(moves))
move_to_idx = {v: k for k, v in idx_to_move.items()}

variations = ["", "'", "2"]
idx_to_variations = dict(enumerate(variations))
variations_to_idx = {v: k for k, v in idx_to_variations.items()}

def GenerateScramble(lenght):
    scramble = []

    if lenght == 0: 
        return scramble
    
    # First move
    move = random.randint(0, len(moves))
    var = random.randint(0, len(variations))
    fullmove = (move, var)
    scramble.append(fullmove)

    if lenght == 1:
        return scramble
    
    # Second move
    while True:
        move = random.randint(0, len(moves))
        # Moving the same side twice is invalid
        if move == scramble[0][0]:
            continue
        
        var = random.randint(0, len(variations))
        fullmove = (move, var)
        scramble.append(fullmove)
        break


    # All other moves
    i = 1
    while i < lenght:
        i += 1
        #print("Current itr: ", i, "(scramble is ", ScrambleToString(scramble), ")")
        move = random.randint(0, len(moves))

        # Moving the same side twice is invalid
        if move == scramble[i - 1][0]:
            #print("first case: ", idx_to_move[move])
            i -= 1
            continue

        # Moving parallel sides thice is invalid
        new_move_type = move // 2
        latest_move_type = scramble[i - 1][0] // 2
        prev_move_type = scramble[i - 2][0] // 2
        if latest_move_type == prev_move_type and new_move_type == latest_move_type:
            #print("second case: ", idx_to_move[move])
            i -= 1
            continue

        var = random.randint(0, len(variations))
        fullmove = (move, var)
        scramble.append(fullmove)
    
    return scramble


def ScrambleToString(scramble):
    string = ""
    for move, var in scramble:
        string = string + idx_to_move[move] + idx_to_variations[var] + " "
    return string[:-1]


for i in range(50):
    scr = GenerateScramble(i if i < 20 else 20)
    print(ScrambleToString(scr))
        
        
