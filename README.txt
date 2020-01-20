Data Science Project: CCNN
-------------------------------------------------------------------------------
1.Requirements

This project requires:
 - Numpy
 - Scikit-learn
 - Pytorch
 - Orion

Orion is only needed if you wish to reproduce the parameter search
(with the option --hunt for main.py or direclty make hunt).
If you wish to do so, you need to have Orion installed with a database set up.
See orion.readthedocs.io for more information

-------------------------------------------------------------------------------
2. Makefile

- make unittests: 
    runs the unit tests in utests
    essentially makes sure everything runs witjout errors

- make hunt:
    runs the hyperparameter search with 3 workers
    currently it is configured for CCNN-3 and m=500
    (takes time)

- make debug_hunt:
    runs a debug hunt to check if the db is correclty set up

- make quick_test:
    runs a quick ccnn training to check if everything is correclty set up

- make test:
    runs a ccnn training to reproduce the plots
    (takes times)

- make test_cnn:
    runs a cnn training to reproduce the plots

-------------------------------------------------------------------------------
3. Project structure

- logger.py:
    very simple logger system for training

- kernels.py:
    a wrapper around sklearn.kernel_approximation tailored for our needs

- layers.py:
    main file: defined CCNNLinearLayer, CCNNLayer and CCNN classes

- cnn.py:
    defines CNN

- utests.py:
    some basic unit tests

- main.py:
    launches the experiment
    read parameters from the command line

-------------------------------------------------------------------------------
4. Branches

- master: most recent
    this implementation tries not to store anything in memory by computing
    everything on the fly, but slow

- in_memory:
    previous implementation, follows the CCNN algorithm naively
    fast but takes a LOT of memory

(the other branches are useless)

-------------------------------------------------------------------------------
5. Remark on the reprsentation of images

    The notations generally follow the article except on one point:
    In the article the spatial representations are "flattened", i.e. images
    and patches are represented by one dimensional vectors. However,
    that going back and forth between 1D and 2D representations (for conv.
    and pooling) induced tricky bugs.
    Therefore, 2D data now stays 2D but for instance A, which is a matrix in
    the article, is now a 4D tensor.

