.. Sympyle documentation master file, created by
   sphinx-quickstart on Fri Jul 13 16:23:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sympyle
*******

.. rubric:: Simple Symbolic Graphs in Python

.. toctree::
   :maxdepth: 2
   :caption: Basics
   :hidden:

   Overview <self>
   Installation <installation>

   :caption: Tutorials
   Logistic Regression <tutorials/Logistic-Regression.ipynb>

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   Contributing <contributing>

.. contents::
   :local:
   :depth: 2


.. automodule:: sympyle.Ops
   :members:


About
#####

Sympyle is a Python library to demonstrate the inner workings of Computational
Graphs. Computational Graphs are used by highly optimised computational
frameworks like `Tensorflow <https://tensorflow.org>`_ and
`Pytorch <https://pytorch.org>`_.

However, these frameworks make several assumptions and optimisations in order
to optimise for speed and memory. This often makes it harder to understand
the inner workings of how these libraries work.

What Sympyle is
---------------

Sympyle is a simplified model library to demonstrate the working of
computational graphs, and how
`backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_
works on arbitrary 'networks'.

Each "operation" will contain detailed documentation on the derivation, and
whenever possible, additional resources to better understand it.



What Sympyle is Not
-------------------

Sympyle is not a production ready library. It should not be thought of as
a competitor to larger frameworks like ``Tensorflow``.

It will also likely not implement all the ops that other libraries use.


