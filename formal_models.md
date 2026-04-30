### Formal Model

Both puzzles are characterized by ***partially observable, deterministic finite labeled transition systems***. Learn more about their causal structure below.

## Puzzle A

**Latent state:** `prev_color`

**Locus:** endogenous (generated from within the agent)

<details>
  <summary>Transition Function</summary>
<br>  
Puzzle A is formally characterized as a partially observable, 
deterministic finite labeled transition system $\mathcal{M} = (S, A, \delta, s_0, F)$:

- **State space:** $S = \text{Grid} \times \text{Colors} \times \text{Colors}$, 
  encoding position and a two-step color history $(pos, c, c')$, 
  where $c'$ (*previous color*) is latent
- **Actions:** $A = \{\uparrow, \downarrow, \leftarrow, \rightarrow\}$
- **Transition:** entering a color tile $k$ updates $(pos, c, c') \rightarrow (pos', k, c)$ <br>
  (stepping on a color tile shifts `prev_color` ← `current_color`, `current_color` ← `tile_color`)
- **Win:** $\text{adjacent}(pos, goal) \wedge c = c_{goal} \wedge c' = \text{gray}$ <br>
  (adjacent to goal AND `current_color` = `goal_color` AND `prev_color` = gray)
</details>

### Pseudo Code:
```python
class MechanicsA:
    def __init__(level):
        avatar_color_history = [level.starting_color]  # predetermined starting color per level
        goal_tile = level.goal_tile  # fixed, does NOT change avatar color
    
    on_input(direction):
        if direction in (UP, DOWN, LEFT, RIGHT):
            move(avatar, direction)  # no movement restraints
    
    on_avatar_enter(tile):
        if FX_tile.type == 'color_changing':
            avatar_color_history.append(tile.color)  # avatar takes on peripheral tile's fixed color    

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

## Puzzle B

**Latent state:** `current_dimension`

**Locus:** exogenous (generated from outside the agent, i.e. environment)

<details>
  <summary>Transition Function</summary>

Puzzle B is formally characterized as a partially observable, deterministic finite labeled transition system M = (S, A, δ, s₀, F):

- **State:** $S = \text{Grid} \times \text{Colors} \times \text{Axes} \times \text{Dimensions} \times \text{TileStates}$,
  where $current\_dimension \in \{D1, D2\}$ is latent
- **Actions:** $A = \{\uparrow, \downarrow, \leftarrow, \rightarrow\}$, filtered by `allowed_axes`
- **Transition:** dotted frame tiles apply $f$($tile$, *current_dimension*) ∈ {direction_change, color_change,empty}</span>
  <br>
  (the dotted frames apply a dimension-dependent function in {direction_change, color_change, empty})
- **Win:** entering_goal($pos$)∧color=c<sub>$goal$</sub>
  <br>​
  (`entering_goal(pos)` AND `avatar_color` = `goal_color`)
</details>

### Pseudo Code:

```python
class MechanicsB:
    def __init__(level):
        allowed_axes = level.starting_axes          # horizontal or vertical, predetermined per level
        current_dimension = level.starting_dimension  # D1 or D2, predetermined per level
    
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
            if FX_tile.type == 'dimension_toggle':
                if current_dimension == 'D1':
                    current_dimension = 'D2'
                elif current_dimension == 'D2':
                    current_dimension = 'D1'
            
            elif FX_tile.type == 'dotted_frame':
                FX_tile.current_function = FX_tile.get_function(current_dimension)  # lookup function in current dimension

                if FX_tile.current_function == 'direction_change':
                    if allowed_axes == 'horizontal':
                        allowed_axes = 'vertical'
                    elif allowed_axes == 'vertical':
                        allowed_axes = 'horizontal'
                
                elif FX_tile.current_function == 'color_change':
                    avatar.color = FX_tile.current_color
                    FX_tile.current_color_index = (FX_tile.current_color_index + 1) % len(FX_tile.color_cycle)
                    FX_tile.current_color = FX_tile.color_cycle[FX_tile.current_color_index]
                
                elif FX_tile.current_function == 'empty':
                    return  # tile does not exist in this dimension; not visible to player

def WinCondition:
    while piece_moving:
        if entering_goal_box:
            if avatar_color == goal_frame_color:
                return level_complete()
            else:
                return

```
