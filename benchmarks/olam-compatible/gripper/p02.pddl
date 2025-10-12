(define (problem gripper-p02)
  (:domain gripper)

  (:objects
    room-a room-b room-c - room
    ball1 ball2 ball3 - ball
    left right - gripper)

  (:init
    (at-robby room-a)
    (free left)
    (free right)
    (at ball1 room-a)
    (at ball2 room-a)
    (at ball3 room-a))

  (:goal
    (and (at ball1 room-b)
         (at ball2 room-c)
         (at ball3 room-b))))
