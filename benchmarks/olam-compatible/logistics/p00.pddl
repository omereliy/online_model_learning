(define (problem logistics-p00)
  (:domain logistics)

  (:objects
    pkg1 - package
    truck1 - truck
    plane1 - airplane
    city1 - city
    loc1-1 loc1-2 - location
    apt1 - airport)

  (:init
    ;; City structure
    (in-city loc1-1 city1)
    (in-city loc1-2 city1)
    (in-city apt1 city1)

    ;; Initial positions
    (at truck1 loc1-1)
    (at plane1 apt1)
    (at-package pkg1 loc1-1))

  (:goal
    (and (at-package pkg1 loc1-2))))
