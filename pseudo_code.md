## Game A Pseudo Code:

```python
class MechanicsA:
    def __init__(level):
        avatar_color_history = [level.starting_color]  # pre-determined starting color per level
        goal_tile = level.goal_tile  # fixed, does NOT change avatar color
    
    on_input(direction):
        if direction in (UP, DOWN, LEFT, RIGHT):
            move(avatar, direction)  # no movement restraints
    
    on_avatar_enter(tile):
        if FX_tile.type == 'color_changing':
            avatar_color_history.append(tile.color)  # avatar takes on that peripheral tile's fixed color
    

def WinCondition: 
    adjacent_tiles = [above(goal_tile), below(goal_tile), left_of(goal_tile), right_of(goal_tile)]
        
    if avatar in adjacent_tiles:
        if avatar_color_history[-1] == goal_tile.color:
            if avatar_color_history[-2] == GRAY:
                return level_complete()
            else:
                return
```

<br>

## Game B Pseudo Code:

```python
class MechanicsB:
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
            if FX_tile.type == 'dotted_frame':

                if FX_tile.current_function == 'direction_change':
                    if allowed_axes == 'horizontal':
                        allowed_axes = 'vertical'
                    elif allowed_axes == 'vertical':
                        allowed_axes = 'horizontal'
                
                elif FX_tile.current_function == 'color_change':
                    avatar.color = FX_tile.current_color
                    FX_tile.current_color_index = (FX_tile.current_color_index + 1) % len(FX_tile.color_cycle)
                    FX_tile.current_color = FX_tile.color_cycle[FX_tile.current_color_index]
    
def WinCondition:
    while piece_moving:
        if entering_goal_box:
            if avatar_color == goal_frame_color:
                return level_complete()
            else:
                return

```