.. Sympyle documentation master file, created by
   sphinx-quickstart on Fri Jul 13 16:23:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sympyle
*******

.. rubric:: Simple Symbolic Graphs in Python

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Basics

   Overview <self>

.. contents::
   :local:
   :depth: 1






About
=====

Sympyle is a Python library to demonstrate the inner workings of Computational
Graphs. Computational Graphs are used by highly optimised computational
frameworks like `Tensorflow <https://tensorflow.org>`_ and
`Pytorch <https://pytorch.org>`_.

However, these frameworks make several assumptions and optimisations in order
to optimise for speed and memory. This often makes it harder to understand
the inner workings of how these libraries work.

Sympyle is a simplified model library to demonstrate the working of
computational graphs, and how
`backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_
works on arbitrary 'networks'.



