# Deep preintegration
This was an old idea to improve IMU-only dead-reckoning. Preintegration is a well-known
technique that isolates a state change over an arbitary long duration into 
state-independent "relative motion increment" (RMI) terms. In theory, these state-independent terms should be learnable as a function of the IMU measurements only. 

Hence we train a model to learn the RMIs. However, it seems that a simple affine model 

``` math 
\mathbf{u}_{calib} = \mathbf{A} \mathbf{u}_{IMU} + \mathbf{b}
```

is able to achieve 99% of the improvement in terms of error reduction. More
sophisticated models have largely diminishing returns.