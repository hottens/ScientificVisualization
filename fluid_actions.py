import enum


class Action(enum.Enum):
    COLOR_DIR = 1
    COLOR_MAG_BLACK = 2
    COLOR_MAG_TWOTONE = 3
    COLOR_MAG_RAINBOW = 4
    MAG_DIR = 5
    VEC_SCALE_UP = 6
    VEC_SCALE_DOWN = 7
    VISC_UP = 8
    VISC_DOWN = 9
    DRAW_SMOKE = 10
    DRAW_CONES = 11
    DRAW_ARROWS = 12
    SCALAR_COLOR_BLACK = 13
    SCALAR_COLOR_TWOTONE = 14
    SCALAR_COLOR_RAINBOW = 15
    COLORMAP_TYPE_DENSITY = 16
    COLORMAP_TYPE_VELOCITY = 17
    COLORMAP_TYPE_FORCES = 18
    FREEZE = 19
    CLAMP_COLOR_MIN_UP = 20
    CLAMP_COLOR_MIN_DOWN = 21
    CLAMP_COLOR_MAX_UP = 22
    CLAMP_COLOR_MAX_DOWN = 23
    CHANGE_HUE = 24
    CHANGE_LEVELS = 25
    SCALE_UP = 26
    SCALE_DOWN = 27
    DT_UP = 28
    DT_DOWN = 29

    COLOR_MAG_CHANGE = 30
    GLYPH_CHANGE = 31
    SCALAR_COLOR_CHANGE = 32
    COLORMAP_CHANGE = 33

    GLYPH_UP = 34
    GLYPH_DOWN = 35
    GLYPH_CHANGE_N = 36

    QUIT = 37

    SET_SCALE_FIELD = 38
    SET_DT = 39
    SET_COLOR_MAG = 40

    CHANGE_ISO_COL = 41
    SET_ISO_MIN = 42
    SET_ISO_MAX = 43
    SET_ISO_N = 44
    COLOR_ISO_BLACK = 45
    COLOR_ISO_RAINBOW = 46
    COLOR_ISO_TWOTONE = 47
    COLOR_ISO_WHITE = 48


    SET_SCALE_ISO = 49
    SET_SCALE_VECTOR = 50

    SET_NLEVELS_FIELD = 51
    SET_NLEVELS_ISO = 52
    SET_NLEVELS_VECTOR = 53

    DRAW_VECS = 54
    DRAW_ISO = 55

    COLORMAP_TYPE_DIVERGENCE = 56

    THREEDIM_ON_OFF = 57
    HEIGHTFACTOR = 58
    HEIGHTSCALE = 59
