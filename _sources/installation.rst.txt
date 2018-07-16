Installation
************

The latest stable version can be downloaded using pip::

	pip install sympyle




Dependency Stack
################
Sympyle is designed to have a very low dependency stack.
The core functionality requires only `numpy <numpy.org>`_.

Enabling Graphing functionality
###############################
By default, ``pygraphviz`` , which is required for drawing the computation
graph is not installed. It is required if graphing functionality is needed.

``pygraphviz`` requires ``graphviz`` to be already installed.

After installing ``graphviz``, ``pygraphviz`` can be installed using pip::

	pip install pygraphviz


.. note:: For mac users, giving pip the location of graphviz is required.
	If graphviz is installed using brew, this may be done using::

		pip install pygraphviz \
		--install-option="--include-path=/usr/local/include/graphviz/" \
		--install-option="--library-path=/usr/local/lib/graphviz"
