.PHONY: unittests
unittests:
	python utests.py

.PHONY: hunt
NAME=MNIST_lazy_500
MAX_TRIALS=2
CMD=./main.py --hunt --lr~'loguniform(1e-5, 1e-2)' --gamma~'loguniform(0.02, 2)' --approx_m 500 --epoch 10
hunt:
	orion hunt -n $(NAME) --worker-trials $(MAX_TRIALS) $(CMD) &
	orion hunt -n $(NAME) --worker-trials $(MAX_TRIALS) $(CMD) &
	orion hunt -n $(NAME) --worker-trials $(MAX_TRIALS) $(CMD) &
debug_hunt:
	orion --debug hunt -n mnist2 --max-trials 2 --pool-size 2 ./main.py --hunt --lr~'loguniform(1e-5, 1.0)' --approx_m 2 --epoch 1

test:
	./main.py --eval_all --lr 0.0005 --approx_m 32 --epoch 5
