(define (domain logistics)
  (:requirements :strips :typing)
  (:types
    vehicle location package - object
    truck airplane - vehicle
    airport city - location)

  (:predicates
    (at ?v - vehicle ?l - location)
    (in ?p - package ?v - vehicle)
    (at-package ?p - package ?l - location)
    (in-city ?l - location ?c - city))

  (:action load-truck
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (at-package ?p ?l)
                       (at ?t ?l))
    :effect (and (in ?p ?t)
                 (not (at-package ?p ?l))))

  (:action unload-truck
    :parameters (?p - package ?t - truck ?l - location)
    :precondition (and (in ?p ?t)
                       (at ?t ?l))
    :effect (and (at-package ?p ?l)
                 (not (in ?p ?t))))

  (:action drive-truck
    :parameters (?t - truck ?from ?to - location ?c - city)
    :precondition (and (at ?t ?from)
                       (in-city ?from ?c)
                       (in-city ?to ?c))
    :effect (and (at ?t ?to)
                 (not (at ?t ?from))))

  (:action load-airplane
    :parameters (?p - package ?a - airplane ?l - airport)
    :precondition (and (at-package ?p ?l)
                       (at ?a ?l))
    :effect (and (in ?p ?a)
                 (not (at-package ?p ?l))))

  (:action unload-airplane
    :parameters (?p - package ?a - airplane ?l - airport)
    :precondition (and (in ?p ?a)
                       (at ?a ?l))
    :effect (and (at-package ?p ?l)
                 (not (in ?p ?a))))

  (:action fly-airplane
    :parameters (?a - airplane ?from ?to - airport)
    :precondition (at ?a ?from)
    :effect (and (at ?a ?to)
                 (not (at ?a ?from)))))