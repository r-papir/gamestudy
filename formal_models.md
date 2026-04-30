### Formal Model

Both puzzles are characterized by ***partially observable, deterministic finite labeled transition systems***.

## Puzzle A

+ **Latent state:** `prev_color`
+ **Locus:** endogenous (generated from within the agent)

### Transition Function:
We characterize Puzzle A as a partially observable, deterministic finite labeled transition system $\mathcal{M} = (S, A, \delta, s_0, F)$, where:

$$S = \text{Grid} \times \text{Colors} \times \text{Colors}$$

encodes avatar position and a two-step color history $(pos, c, c')$, with $c$ denoting the current color and $c'$ the previous color (latent).

The action space is $A = \{\uparrow, \downarrow, \leftarrow, \rightarrow\}$, with no movement constraints.

The transition function $\delta: S \times A \rightarrow S$ is defined as:

$$\delta((pos, c, c'), a) = \begin{cases} (pos', k, c) & \text{if } \text{color\_tile}(pos') = k \\ (pos', c, c') & \text{otherwise} \end{cases}$$

where $pos' = \text{move}(pos, a)$.

The initial state is $s_0 = (start\_pos, c_0, \text{gray})$, where $c_0$ is the level-specific starting color.

The set of winning states is:

$$F = \{(pos, c, c') \in S \mid \text{adjacent}(pos, goal\_pos) \wedge c = c_{goal} \wedge c' = \text{gray}\}$$


Actions:
A = {↑, ↓, ←, →}

Transition function δ: S × A → S:
δ((pos, current_color, prev_color), a) =

  let pos' = move(pos, a)

  if color_changing_tile(pos'):
      let c = tile_color(pos')
      return (pos', c, current_color)   % shift color history forward

  return (pos', current_color, prev_color)   % no color change

Win condition F ⊆ S:
F = {(pos, current_color, prev_color) | adjacent(pos, goal_pos) ∧ current_color = goal_color ∧ prev_color = gray}

Initial state:
s₀ = (start_pos, starting_color, gray)

Full specification tuple:
M = (S, A, δ, s₀, F)



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
+ **Latent state:** `current_dimension`
+ **Locus:** exogenous (generated from outside the agent, i.e. environment)

### Pseudo Code: Puzzle B

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
