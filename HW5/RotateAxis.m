function R = RotateAxis( h, theta )
%UNTITLED5 �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ

cos_f = cos(-theta);
sin_f = sin(-theta);

R = [cos_f+h(1)^2*(1-cos_f), h(1)*h(2)*(1-cos_f)-h(3)*sin_f, h(1)*h(3)*(1-cos_f)+h(2)*sin_f;
    h(2)*h(1)*(1-cos_f)+h(3)*sin_f, cos_f+h(2)^2*(1-cos_f), h(2)*h(3)*(1-cos_f)-h(1)*sin_f;
    h(3)*h(1)*(1-cos_f)-h(2)*sin_f, h(3)*h(2)*(1-cos_f)+h(1)*sin_f, cos_f+h(3)^2*(1-cos_f)];

end

