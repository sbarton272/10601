function [] = feedback(number, correct)
    if correct
        disp(sprintf('%d is correct', number));
    else
        disp(sprintf('%d is incorrect', number));
    end
end
