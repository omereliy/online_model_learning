(define (domain depot)
(:requirements :strips :typing)
(:types
    place locatable - object
    depot distributor - place
    truck hoist surface - locatable
    pallet crate - surface)

(:predicates
    (at ?x0 - locatable ?x1 - place)
    (on ?x0 - crate ?x1 - surface)
    (in ?x0 - crate ?x1 - truck)
    (lifting ?x0 - hoist ?x1 - crate)
    (available ?x0 - hoist)
    (clear ?x0 - surface))

(:action drive
  :parameters (?x0 - truck ?x1 - place ?x2 - place)
  :precondition (and (at ?x0 ?x1))
  :effect (and (not (at ?x0 ?x1)) (at ?x0 ?x2)))

(:action lift
  :parameters (?x0 - hoist ?x1 - crate ?x2 - surface ?x3 - place)
  :precondition (and (at ?x0 ?x3) (available ?x0) (at ?x1 ?x3) (on ?x1 ?x2) (clear ?x1))
  :effect (and (not (at ?x1 ?x3)) (lifting ?x0 ?x1) (not (clear ?x1)) (not (available ?x0)) (clear ?x2) (not (on ?x1 ?x2))))

(:action drop
  :parameters (?x0 - hoist ?x1 - crate ?x2 - surface ?x3 - place)
  :precondition (and (at ?x0 ?x3) (at ?x2 ?x3) (clear ?x2) (lifting ?x0 ?x1))
  :effect (and (available ?x0) (not (lifting ?x0 ?x1)) (at ?x1 ?x3) (not (clear ?x2)) (clear ?x1) (on ?x1 ?x2)))

(:action load
  :parameters (?x0 - hoist ?x1 - crate ?x2 - truck ?x3 - place)
  :precondition (and (at ?x0 ?x3) (at ?x2 ?x3) (lifting ?x0 ?x1))
  :effect (and (not (lifting ?x0 ?x1)) (in ?x1 ?x2) (available ?x0)))

(:action unload
  :parameters (?x0 - hoist ?x1 - crate ?x2 - truck ?x3 - place)
  :precondition (and (at ?x0 ?x3) (at ?x2 ?x3) (available ?x0) (in ?x1 ?x2))
  :effect (and (not (in ?x1 ?x2)) (not (available ?x0)) (lifting ?x0 ?x1)))
)
