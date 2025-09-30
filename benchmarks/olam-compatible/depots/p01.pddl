(define (problem depotprob3909) (:domain depot)
(:objects
	depot0 - depot
	distributor0 distributor1 - distributor
	truck0 truck1 - truck
	pallet0 pallet1 pallet2 - pallet
	crate0 crate1 crate2 crate3 crate4 - crate
	hoist0 hoist1 hoist2 - hoist)
(:init
	(at pallet0 depot0)
	(clear pallet0)
	(at pallet1 distributor0)
	(clear pallet1)
	(at pallet2 distributor1)
	(clear crate4)
	(at truck0 distributor0)
	(at truck1 distributor1)
	(at hoist0 depot0)
	(available hoist0)
	(at hoist1 distributor0)
	(available hoist1)
	(at hoist2 distributor1)
	(available hoist2)
	(at crate0 distributor1)
	(on crate0 pallet2)
	(at crate1 distributor1)
	(on crate1 crate0)
	(at crate2 distributor1)
	(on crate2 crate1)
	(at crate3 distributor1)
	(on crate3 crate2)
	(at crate4 distributor1)
	(on crate4 crate3)
)

(:goal (and
		(on crate0 crate2)
		(on crate1 pallet0)
		(on crate2 crate3)
		(on crate3 pallet2)
		(on crate4 pallet1)
	)
))
