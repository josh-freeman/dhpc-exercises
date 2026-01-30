bool at_goal[4];
bool torch_at_goal=false;// tells you which direction the next crossing goes.
inline max(a, b, result) {
    if
    :: (a >= b) -> result = a
    :: (a <  b) -> result = b
    fi
}

init{
    int elapsed = 0;
    int times[4] = {5, 10, 20, 25};
    int temp;
    do
    :: elapsed <= 60 && ! (at_goal[0] && at_goal[1] && at_goal[2] && at_goal[3]) ->
        if
        :: !torch_at_goal -> // 1 or two people go from start to goal
            if
            :: (!at_goal[0] && !at_goal[1]) -> at_goal[0] = true; at_goal[1] = true; max(times[0], times[1], temp); elapsed = elapsed + temp;
            :: (!at_goal[0] && !at_goal[2]) -> at_goal[0] = true; at_goal[2] = true; max(times[0], times[2], temp); elapsed = elapsed + temp;
            :: (!at_goal[0] && !at_goal[3]) -> at_goal[0] = true; at_goal[3] = true; max(times[0], times[3], temp); elapsed = elapsed + temp;
            :: (!at_goal[1] && !at_goal[2]) -> at_goal[1] = true; at_goal[2] = true; max(times[1], times[2], temp); elapsed = elapsed + temp;
            :: (!at_goal[1] && !at_goal[3]) -> at_goal[1] = true; at_goal[3] = true; max(times[1], times[3], temp); elapsed = elapsed + temp;
            :: (!at_goal[2] && !at_goal[3]) -> at_goal[2] = true; at_goal[3] = true; max(times[2], times[3], temp); elapsed = elapsed + temp;
            :: (!at_goal[0]) -> at_goal[0] = true; elapsed = elapsed + times[0];
            :: (!at_goal[1]) -> at_goal[1] = true; elapsed = elapsed + times[1];
            :: (!at_goal[2]) -> at_goal[2] = true; elapsed = elapsed + times[2];
            :: (!at_goal[3]) -> at_goal[3] = true; elapsed = elapsed + times[3];
            fi;
        :: torch_at_goal->  // 1 person goes back to start from goal
            if 
            :: (at_goal[0]) -> at_goal[0] = false; elapsed = elapsed + times[0];
            :: (at_goal[1]) -> at_goal[1] = false; elapsed = elapsed + times[1];
            :: (at_goal[2]) -> at_goal[2] = false; elapsed = elapsed + times[2];
            :: (at_goal[3]) -> at_goal[3] = false; elapsed = elapsed + times[3];
            fi;
            
        fi;
        torch_at_goal = !torch_at_goal;
    :: elapsed > 60 || (at_goal[0] && at_goal[1] && at_goal[2] && at_goal[3]) -> break;
    od;

    assert(!(at_goal[0] && at_goal[1] && at_goal[2] && at_goal[3]) || elapsed > 60);
}