

(define (problem BW-rand-7)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7  - block)
(:init
(handempty)
(on b1 b7)
(on b2 b5)
(ontable b3)
(on b4 b6)
(on b5 b4)
(ontable b6)
(on b7 b2)
(clear b1)
(clear b3)
)
(:goal
(and
(on b3 b2)
(on b6 b4)
(on b7 b6))
)
)


