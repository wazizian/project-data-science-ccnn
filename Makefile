.PHONY: unittests
unittests:
	python utests.py

NAME=MNIST2
hunt:
	orion hunt -n $(NAME) --max-trials 10 ./main.py --hunt --lr~'loguniform(1e-5, 1.0)' --gamma~'loguniform(0.02, 2)' --approx_m 32 --epoch 5

debug_hunt:
	orion --debug hunt -n mnist2 --max-trials 2 ./main.py --hunt --lr~'loguniform(1e-5, 1.0)' --approx_m 2 --epoch 1

test:
	./main.py --eval_all --lr 0.00045333440803464417 --approx_m 32 --epoch 2 --cnn
