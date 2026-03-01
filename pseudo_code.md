## Game A Pseudo Code:

```python
def WinCondition:
    if avatar_color_history[-1] == goal_color:
    if avatar_color_history[-2] == GRAY:
	    return level_complete()
    else:
	    return
```

<br>

## Game B Pseudo Code:

```python
class Mechanics:
    def __init__(level):
        allowed_axes = level.starting_axes  # 'horizontal' or 'vertical', pre-determined per level
    
    while avatar_moving:
        on_input(direction):
            
            if allowed_axes == 'horizontal':
                if direction in (LEFT, RIGHT):
                    move(avatar, direction)
                else:
                    ignore_input()
            
            elif allowed_axes == 'vertical':
                if direction in (UP, DOWN):
                    move(avatar, direction)
                else:
                    ignore_input()
        
        on_FX_tile_enter(tile):
            if tile.type == 'dotted_frame':
                if tile.current_function == 'direction_change':
                    if allowed_axes == 'horizontal':
                        allowed_axes = 'vertical'
                    elif allowed_axes == 'vertical':
                        allowed_axes = 'horizontal'
                elif tile.current_function == 'color_change':
                    avatar.color = tile.current_color

class ColorChangingTile:
    def __init__(level):
        color_cycle = level.assigned_color_cycle  # pre-determined color sequence per level
        current_color_index = 0
        current_color = color_cycle[0]
    
    on_avatar_enter(tile):
        avatar.color = current_color
        current_color_index = (current_color_index + 1) % len(color_cycle)
        current_color = color_cycle[current_color_index]


while piece_moving:     # win condition
    if entering_goal_box:
        if avatar_color == goal_frame_color:
            return level_complete()
        else:
            return
```