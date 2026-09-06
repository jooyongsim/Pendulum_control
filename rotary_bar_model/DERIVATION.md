# Derivation: uniform-bar rotary pendulum

## 1. Rotational equivalent of Newton's second law

For translation,

\[
F=ma.
\]

For rotation about a fixed axis,

\[
\tau=J\alpha=J\ddot q.
\]

Thus torque creates angular acceleration, and the rotational inertia \(J\) plays the role that mass plays in translation.

For a point mass \(m\) at distance \(x\), \(J=mx^2\). For a distributed body,

\[
J=\int x^2\,dm.
\]

For the uniform pendulum bar of total length \(l\), pivoted at one end,

\[
\boxed{J_p=\frac13ml^2},\qquad
\boxed{l_c=\frac l2}.
\]

The gravitational force \(mg\) acts at the center of mass, so the magnitude of the gravitational torque is based on \(l_c=l/2\), not on the full length.

## 2. Coordinates

- \(\phi\): rotary-arm angle about the vertical motor axis.
- \(r\): distance from motor axis to the pendulum pivot.
- \(\vartheta\): global pendulum angle measured from the downward vertical.
- \(\vartheta=0\): suspended/downward equilibrium.
- \(\vartheta=\pi\): inverted/upright equilibrium.

The pendulum is a uniform bar of mass \(m\) and total length \(l\).

## 3. Low-speed rotary-pendulum equation

The full nonlinear model contains velocity-product and velocity-squared terms such as

\[
\dot\phi\dot\vartheta,\qquad \dot\phi^2,\qquad \dot\vartheta^2.
\]

Here we assume the angular velocities are small enough that these terms can be neglected. The pendulum equation then reduces to

\[
\boxed{
\frac13ml^2\ddot\vartheta
+\frac12mrl\cos\vartheta\,\ddot\phi
+\frac12mgl\sin\vartheta=0.
}
\]

Each term has a direct physical meaning:

1. \(\frac13ml^2\ddot\vartheta\): torque required to angularly accelerate the bar.
2. \(\frac12mrl\cos\vartheta\,\ddot\phi\): coupling torque caused by accelerating the arm pivot.
3. \(\frac12mgl\sin\vartheta\): gravitational torque.

Divide by \(J_p=\frac13ml^2\):

\[
\boxed{
\ddot\vartheta
+\frac{3r}{2l}\cos\vartheta\,\ddot\phi
+\frac{3g}{2l}\sin\vartheta=0.
}
\]

Notice that \(m\) cancels. Under the ideal acceleration-input model, the pendulum angle response does not depend on bar mass. Mass still matters for the actual motor torque required to realize \(\ddot\phi\).

Define

\[
a=\frac{3g}{2l},\qquad b=\frac{3r}{2l},\qquad u=\ddot\phi.
\]

## 4. Suspended equilibrium

For the suspended case, define the local angle directly as

\[
\theta=\vartheta,
\]

so \(\theta=0\) is downward. Near zero,

\[
\sin\theta\approx\theta,\qquad \cos\theta\approx1.
\]

Then

\[
\boxed{\ddot\theta+a\theta=-bu.}
\]

Taking the Laplace transform with zero initial conditions,

\[
(s^2+a)\Theta(s)=-bU(s),
\]

therefore

\[
\boxed{P_s(s)=\frac{\Theta(s)}{U(s)}=-\frac{b}{s^2+a}}.
\]

The minus sign expresses the inertial effect: accelerating the pivot in one direction initially tilts the suspended bar in the opposite local direction.

Without damping the poles are

\[
s=\pm j\sqrt a,
\]

so the ideal suspended model oscillates forever. Real bearing friction and air drag add damping.

## 5. Inverted equilibrium

For the inverted case define a local angle around upright:

\[
\vartheta=\pi+\theta,
\]

so again the equilibrium itself is \(\theta=0\). For small \(\theta\),

\[
\sin(\pi+\theta)\approx-\theta,\qquad
\cos(\pi+\theta)\approx-1.
\]

Substituting into the low-speed equation gives

\[
\boxed{\ddot\theta-a\theta=bu.}
\]

Hence

\[
\boxed{P_i(s)=\frac{\Theta(s)}{U(s)}=\frac{b}{s^2-a}}.
\]

The poles are

\[
s=\pm\sqrt a.
\]

One pole is on the positive real axis, so upright is open-loop unstable.

## 6. Numerical model for this setup

Using

\[
l=0.235\,\mathrm m,\qquad r=0.14\,\mathrm m,\qquad g=9.81\,\mathrm{m/s^2},
\]

we obtain

\[
a=\frac{3g}{2l}=62.6170213\,\mathrm{s^{-2}},
\]

\[
b=\frac{3r}{2l}=0.8936170.
\]

Therefore

### Suspended

\[
\boxed{\ddot\theta+62.6170\theta=-0.893617u}
\]

and

\[
\boxed{P_s(s)=-\frac{0.893617}{s^2+62.6170}}.
\]

### Inverted

\[
\boxed{\ddot\theta-62.6170\theta=0.893617u}
\]

and

\[
\boxed{P_i(s)=\frac{0.893617}{s^2-62.6170}}.
\]

## 7. Relation to the earlier angle-input transfer function

If the input is arm angle \(\Phi(s)\) instead of arm angular acceleration, then

\[
U(s)=s^2\Phi(s)
\]

for zero initial conditions. Therefore the angle-input transfer functions acquire an additional \(s^2\) in the numerator. In this folder the actuator input is explicitly defined as \(u=\ddot\phi\), so the plant numerator is simply \(\pm b\).

## 8. PID stabilization idea for upright

For the inverted plant

\[
P_i(s)=\frac{b}{s^2-a},
\]

use negative feedback with

\[
e=\theta_{ref}-\theta,
\]

\[
u=K_pe+K_i\int e\,dt+K_d\dot e.
\]

The closed-loop characteristic polynomial is

\[
\boxed{s^3+bK_ds^2+(bK_p-a)s+bK_i=0.}
\]

A necessary proportional-gain condition for the \(s\) coefficient to become positive is

\[
K_p>\frac{a}{b}=\frac{g}{r}.
\]

For \(r=0.14\,\mathrm m\),

\[
\frac gr\approx70.07.
\]

The PID notebook chooses desired stable poles and solves the polynomial coefficient equations for \(K_p,K_i,K_d\), then checks the result in time-domain simulation with an acceleration-command limit.
