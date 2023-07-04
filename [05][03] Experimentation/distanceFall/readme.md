## Distance Falloff
### Description
The systems boundary is a single plane and the transition between is happening sharply. This function is used to create a smooth transition between two systems.

### How it works
The function is based on inverse square law, with modified parameters. The function is defined as:
$$
f(x) = \frac{0.5}{1 + (x*10)^2}
$$
Where $x$ is the distance from the the position of the system and the dividing plane, and $b$ is a parameter that controls the falloff. 
The function is normalized so that the maximum value is 0.5. On the dividing plane the contributions from the two systems are equal.


