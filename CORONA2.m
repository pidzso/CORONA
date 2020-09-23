syms a b c d e ff(t) ss(t) aa(t) ii(t) rr(t) dd(t) hh mm

% Solution using dsolve
%ff(t) = ((1 - hh) * aa(t) * (1 - hh) * (1 - mm)) / (ss(t) + aa(t) + rr(t));
ff(t) = aa(t) / (ss(t) + aa(t) + rr(t));
A = [-ff(t), 0,    0,    e, 0; 
     ff(t),  -a-c, 0,    0, 0; 
     0,     a,    -b-d, 0,  0; 
     0,     c,    b,    -e, 0; 
     0,     0,    d,    0,  0;];
X = [ss; aa; ii; rr; dd];
dsolve(diff(X) == A*X, X(0)==[0.9;0.1;0;0;0]);

% Solution using ode
f = @(t,x) [-x(2)/(x(1)+x(2)+x(3))*x(1)+e*x(4);-x(2)/(x(1)+x(2)+x(3))*x(1)-(a+c)*x(2);a*x(2)-(b+d)*x(3);c*x(2)+b*x(3)-e*x(4);d*x(3)];
[t,xa] = ode45(f,[0 99],[0.9 0.1 0 0 0]);

% plot second function
%plot(t,xa(:,2))
%title('aa(t)')
%xlabel('t'), ylabel('aa')

% Solution using bvp
%init = ;
%bvp4c(diff(X) == A*X, X(0)==[0.9;0.1;0;0;0], init)
%bvp5c(diff(X) == A*X, X(0)==[0.9;0.1;0;0;0], init)
