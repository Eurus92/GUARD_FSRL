Contributing
=============


Install Develop Version
-----------------------

To install FSRL in an "editable" mode, run

.. code-block:: bash

    $ pip install -e ".[dev]"

in the main directory. This installation is removable by

.. code-block:: bash

    $ python setup.py develop --uninstall


PEP8 Code Style Check and Code Formatter
----------------------------------------

Please set up pre-commit by running

.. code-block:: bash

    $ pre-commit install

in the main directory. This should make sure that your contribution is properly
formatted before every commit.

We follow PEP8 python code style with flake8. To check, in the main directory, run:

.. code-block:: bash

    $ make lint

We use isort and yapf to format all codes. To format, in the main directory, run:

.. code-block:: bash

    $ make format

To check if formatted correctly, in the main directory, run:

.. code-block:: bash

    $ make check-codestyle


Type Check
----------

We use `mypy <https://github.com/python/mypy/>`_ to check the type annotations. To check, in the main directory, run:

.. code-block:: bash

    $ make mypy


Test Locally
------------

This command will run automatic tests in the main directory

.. code-block:: bash

    $ make pytest


Test by GitHub Actions
----------------------

1. Click the ``Actions`` button in your own repo:

.. image:: _static/images/action1.jpg
    :align: center

2. Click the green button:

.. image:: https://tianshou.readthedocs.io/en/master/_images/action2.jpg
    :align: center

3. You will see ``Actions Enabled.`` on the top of html page.

4. When you push a new commit to your own repo (e.g. ``git push``), it will automatically run the test in this page:

.. image:: https://tianshou.readthedocs.io/en/master/_images/action3.png
    :align: center


Documentation
-------------

Documentations are written under the ``docs/`` directory as ReStructuredText (``.rst``) files. ``index.rst`` is the main page. A Tutorial on ReStructuredText can be found `here <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_.

API References are automatically generated by `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ according to the outlines under ``docs/api/`` and should be modified when any code changes.

To compile documentation into webpage, run

.. code-block:: bash

    $ make doc

The generated webpage is in ``docs/_build`` and can be viewed with browser (http://0.0.0.0:8000/).


Documentation Generation Test
-----------------------------

We have the following three documentation tests:

1. pydocstyle: test all docstring under ``fsrl/``;

2. doc8: test ReStructuredText format;

3. sphinx test: test if there is any error/warning when generating front-end html documentation.

To check, in the main directory, run:

.. code-block:: bash

    $ make check-docstyle
