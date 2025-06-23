function pregunta1()
  clc; clear;
  x=[0 2 4 6];
  y=30*2.^(x/2);
  h=2;
  [a,b,c,d]=trazador_cubico(y,h)

end


function A=tridiagonal(n)
  A=zeros(n);
  A(1,1)=1; A(n,n)=1;
  for i=2:n-1
    A(i,i-1)=1;
    A(i,i)=4;
    A(i,i+1)=1;
  end
end

function w=vector_(y,h)
  n=length(y);
  w=zeros(n,1);
  for i=2:n-1
    w(i)=(6/h^2)*(y(i-1)-2*y(i)+y(i+1))  ;
  end
end

function [a,b,c,d]=trazador_cubico(y,h)
  n=length(y);
  A=tridiagonal(n);
  w=vector_(y,h);
  z=mldivide(A,w);
  a=zeros(n-1,1);
  b=zeros(n-1,1);
  c=zeros(n-1,1);
  d=zeros(n-1,1);
  for j=1:n-1
    a(j)=z(j+1)/(6*h);
    b(j)=z(j)/(6*h);
    c(j)=y(j+1)/h-(h/6)*z(j+1);
    d(j)=y(j)/h-(h/6)*z(j);
  end
end






