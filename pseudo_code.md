## Game A Pseudo Code:

```python
def Win_Condition:
    if avatar_color_history[-1] == goal_color:
    if avatar_color_history[-2] == GRAY:
	    return level_complete()
    else:
	    return
```

<br>

## Game B Pseudo Code:

while avatar_moving:
        on_input(direction):
            
            if allowed_axes == None:
                if direction in (LEFT, RIGHT):
                    allowed_axes = 'horizontal'
                elif direction in (UP, DOWN):
                    allowed_axes = 'vertical'
            
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
        
        on_tile_enter(tile):
            if tile.type == 'dotted_frame':
                if tile.current_function == 'direction_change':
                    if allowed_axes == 'horizontal':
                        allowed_axes = 'vertical'
                    elif allowed_axes == 'vertical':
                        allowed_axes = 'horizontal'
                
                elif tile.current_function == 'color_change':
                    avatar.color = tile.color
                
                # tile appears identical visually regardless of current_function
                # player must track hidden state from prior interactions