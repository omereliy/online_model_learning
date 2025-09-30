(define (problem blocksworld-p01)
  (:domain blocksworld)

  (:objects
    a b c - block
  )

  (:init
    (clear a)
    (on a b)
    (on b c)
    (ontable c)
    (handempty)
  )

  (:goal
    (and (on c b)
         (on b a))
  )
)