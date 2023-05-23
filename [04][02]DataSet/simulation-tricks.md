	
# On Creating the Dataset


## method
    IdealeZahlen 6 months ago | prev | next [–]
        - What kind of ODE solvers are used to simulate chaotic systems? They must be very accurate if even a small error can result in a completely different result.

    j7ake 6 months ago | parent | next [–]
        - You can use the standard Euler method with very small delta T.

    pixelpoet 6 months ago | root | parent | next [–]
        - Respectfully disagree; Euler's method is absolutely terrible because it's unconditionally unstable; far better is to use something like Leapfrog or velocity Verlet (both of which have 2nd order accuracy and better stability, for exactly the same number of derivative evaluations). Euler integration is essentially always the wrong tool for the job.

    karpierz 6 months ago | root | parent | next [–]
        - Would RK4 work for something like this, or does it lack some stability properties?
	
    IIAOPSW 6 months ago | root | parent | next [–]
        - If you want to see the folly of RK4, give it something like a ball bouncing and watch as it bounces slightly higher each time. Subtle at first, but trust me its there. The comment above you is right. If you have conservation of energy and want to keep it that way, use verlet.
        Edit: before anyone calls me out on this, the same trick also works when there's no discontinuity in the force function vis a vis collision with the floor. Planetary motion also drifts out of simple orbits. I just picked the ball bouncing because its a more amusing visual.

    jordigh 6 months ago | root | parent | prev | next [–]
        - RK4 is about as unstable. The backwards Euler method is baby's first stable ODE solver. If you want to make RK4 stable, you have to change it into the implicit RK4 method.
        However, the Lorenz system isn't stiff, which is usually what you use stable solvers for, nor any of the other systems here, I believe. RK4 should be fine, or even normal Euler with a reasonable step size. The chaos you see in these systems is not due to numerical inaccuracy. They're inherently chaotic. That's what makes chaos a subject worth studying.

