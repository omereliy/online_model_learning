(define (problem gripper-p01)
  (:domain gripper)

  (:objects
    room-a room-b - room
    ball1 ball2 - ball
    left right - gripper)

  (:init
    (at-robby room-a)
    (free left)
    (free right)
    (at ball1 room-a)
    (at ball2 room-a))

  (:goal
    (and (at ball1 room-b)
         (at ball2 room-b))))