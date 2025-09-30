(define (problem logistics-p01)
  (:domain logistics)

  (:objects
    pkg1 pkg2 - package
    truck1 - truck
    plane1 - airplane
    city1 city2 - city
    loc1-1 loc1-2 - location
    loc2-1 - location
    apt1 apt2 - airport)

  (:init
    ;; City structure
    (in-city loc1-1 city1)
    (in-city loc1-2 city1)
    (in-city apt1 city1)
    (in-city loc2-1 city2)
    (in-city apt2 city2)

    ;; Initial positions
    (at truck1 loc1-1)
    (at plane1 apt1)
    (at-package pkg1 loc1-1)
    (at-package pkg2 loc1-2))

  (:goal
    (and (at-package pkg1 loc2-1)
         (at-package pkg2 apt2))))