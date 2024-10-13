# NOTE:
Expect baseline repo update
now has SIFT feature and stuff

# IDEAS for Solving
* Method - Exploration: Pose Graph Feature Map

# Environment Information
- Randomly generated maze
- You want to navigate the maze to find a target image

# Exploration Data
- Going around the maze to get images (320x240)
- Consider down-sampling in time for init dev

# Game Control (Baseline Solution)
- WASD
- Q query image
    - gives you a new window called next best view
    - you want to try all the possibilities to find the image closest to your goal
- SPACE goal check-in (and end game)
    - saves game trajectory (game.mpy) stores all actions, time, etc. allows them to replay your run
- ESC to move from exploration phase to navigation phase

# Basline Solution Pipeline
Exploration Phase: Pre-Nav Computation Phase:
Exploration Images -> SIFT Feature -> K-Means Cluster (building visual world/vocab) -> VLAD Descriptor (Structure from Motion (Lec 8)) -> Ball Tree (Structure?)

Navigation Phase
-> Current View given to Ball Tree -> Ball Tree returns Nearest Neighbor of current view
assuming you know image index of the goal image that you can step through to find the goal image

# Dev
- player.py is skeleton code to work off on as a reference
- basline.py is a reference for how to make your implementation

# Manual Use Baseline (Navigation Phase)
- Run, wait for SIFT, now you do exploration manually
- Run around and query images, and you get info on what to do next
    - Don't query deadends
    - Basically you want the Next View ID to get closer to Goal ID
- Baseline has no parameter tuning on stuff like K-Means clustering

