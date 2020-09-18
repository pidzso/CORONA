syms ff(t) ss(t) aa(t) ii(t) rr(t) dd(t) alpha beta gamma delta eps zeta hh mm

ff(t) = zeta * (1 - hh) * aa(t) * (1 - hh) * (1 - mm);

ode1 = diff(ss) == eps * rr - ff * ss;
ode2 = diff(aa) ==  - beta * aa + ff * ss;
ode3 = diff(ii) == alpha * aa - delta * ii - gamma * ii;
ode4 = diff(rr) == beta * aa + gamma * ii-eps * rr;
ode5 = diff(dd) == delta * ii;

odes = [ode1; ode2; ode3; ode4; ode5];

S = dsolve(odes)


syms a b c d e f
A = [-a, 0,    0,    f,  0; 
     a,  -b-d, 0,    0,  0; 
     0,  b,    -c-e, 0,  0; 
     0,  d,    c,    -f, 0; 
     0,  0,    e,    0,  0;];
 
