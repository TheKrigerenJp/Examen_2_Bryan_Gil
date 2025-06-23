function ejemplo_lagrange()
  clc; clear
  pkg load symbolic
  xv=[-2 0 1];
  yv=[0 1 -1];
  p=pol_lagrange(xv,yv); %Simbolico
  p
end

function p=pol_lagrange(xv,yv)
  n=length(xv)-1;
  p=0;
  for k=0:n
    p=p+yv(k+1)*Lk(xv,k);
  end
  p= expand(p); %Simplifica la suma
end

function L=Lk(xv,k)
  syms x
  n=length(xv)-1;
  L=1;
  for j=0:n
    if j~=k
      L=L*(x-xv(j+1))/(xv(k+1)-xv(j+1));
    end
  end
end




