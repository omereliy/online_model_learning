(define (problem gripper-p00)
  (:domain gripper)

  (:objects
    room-a - room
    ball1 - ball
    left right - gripper)

  (:init
    (at-robby room-a)
    (free left)
    (free right)
    (at ball1 room-a))

  (:goal
    (and (at ball1 room-a))))
