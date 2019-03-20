"""
Low-level optimization options.
"""

# Copyright 2019 Ethan I. Schaefer
#
# This file is part of numgeo.
#
# numgeo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# numgeo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with numgeo.  If not, see <https://www.gnu.org/licenses/>.

__version__ = "0.0.1a0"
__author__ = "Ethan I. Schaefer"


###############################################################################
# USER SETTINGS                                                               #
###############################################################################
# Flat arrays with lengths shorter than the value specified below will
# be summed by sum(a.tolist()) (instead of a.sum()).
OPTIMIZE_SUM_CUTOFF = 120
# Flat arrays with lengths shorter than the value specified below will
# be examined by min/max(a.tolist()) (instead of a.min()/max()).
OPTIMIZE_EXTREME_CUTOFF = 70
