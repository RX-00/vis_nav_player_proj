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

# Meeting 10/16/2024
We only have navigation phase to work with (they provide us their exploration data (just a bunch of images saved from moving randomly in the maze)).

Idea from Denis (depends on what TA responds with):

We can generate a pose graph based on the camera matrix to recover rotation and translation. This pose graph can then be solved for shortest optimal path via A* or something like that. Then we can show the nodes of the pose graph referring to the images and manually move around the maze

# Meeting 10/29/2024
Things haven't been working out, so we're falling back on the baseline.Don't forget to download and use the new exploration data.

[] Denis makes a good suggestion that we use the FAST feature detector instead SIFT. Use ORB as the descriptor.

[] Retune the n_clusters for KMeans and leaf_size for the BallTree.

[] Update the report
