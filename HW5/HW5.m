clear all;
 p1 = [-0.5, 0, 2.121320];
 p2 = [0.5, 0, 2.121320];
 p3 = [0.5, -0.707107, 2.828427];
 p4 = [0.5, 0.707107, 2.828427];
 p5 = [1,1,1];

p1_out = [1.363005, -0.427130, 2.339082];
p2_out = [1.748084, 0.437983, 2.017688];
p3_out = [2.636461, 0.184843, 2.400710];
p4_out = [1.4981, -0.8710, 2.8837];

p1_0 = p1-p1;
p2_0 = p2-p1;
p3_0 = p3-p1;
h = cross(p2_0-p1_0, p3_0-p1_0);
h_out = cross(p2_out-p1_out, p3_out-p1_out);

%% Calucate R1, using Jyrki Lahtonen's reflection method
%% ref: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
a=h'; b=h_out';
u = a/norm(a);                     
v = b/norm(b);                      
N = length(u);
A = eye(N);
S = jyrki_reflection(A, v+u );     
R1 = jyrki_reflection( S, v );             

%% Calculate R2, using Rotation matrix from axis and angle 
u_o = R1*(p3-p1)'/norm(R1*(p3-p1)');
v_o = (p3_out-p1_out)/norm(p3_out-p1_out);
theta = acos(dot(u_o,v_o));
R2 = RotateAxis( h_out, theta );

input = p1
p_rigid = R2*R1*(input-p1)'+p1_out';
output = p_rigid'








