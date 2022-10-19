/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.dataset

/**
 * Type of Imagenet dataset labels.
 * Some models are trained on the vanilla ImageNet dataset, which contains 1000 classes,
 * but some models also output additional 'background' class.
 */
public enum class Imagenet {
    V1k,
    V1001;

    public fun labels(zeroIndexed: Boolean = true): Map<Int, String> {
        return when (this) {
            V1k -> if (!zeroIndexed) toOneIndexed(imagenetLabels) else imagenetLabels
            V1001 -> if (!zeroIndexed) toOneIndexed(addBackgroundLabel(imagenetLabels)) else
                addBackgroundLabel(imagenetLabels)
        }
    }

    private fun toOneIndexed(labels: Map<Int, String>): Map<Int, String> {
        val zeroIndexedLabels = mutableMapOf<Int, String>()
        labels.forEach { (key, value) ->
            zeroIndexedLabels[key + 1] = value
        }
        return zeroIndexedLabels
    }

    private fun addBackgroundLabel(labels: Map<Int, String>): Map<Int, String> {
        val labelsWithBackground = mutableMapOf<Int, String>()
        labelsWithBackground[0] = "background"
        labels.forEach { (key, value) ->
            labelsWithBackground[key + 1] = value
        }
        return labelsWithBackground
    }
}

/**
 * Map of Imagenet labels.
 */
public val imagenetLabels: Map<Int, String> = mapOf(
    0 to "tench",
    1 to "goldfish",
    2 to "great_white_shark",
    3 to "tiger_shark",
    4 to "hammerhead",
    5 to "electric_ray",
    6 to "stingray",
    7 to "cock",
    8 to "hen",
    9 to "ostrich",
    10 to "brambling",
    11 to "goldfinch",
    12 to "house_finch",
    13 to "junco",
    14 to "indigo_bunting",
    15 to "robin",
    16 to "bulbul",
    17 to "jay",
    18 to "magpie",
    19 to "chickadee",
    20 to "water_ouzel",
    21 to "kite",
    22 to "bald_eagle",
    23 to "vulture",
    24 to "great_grey_owl",
    25 to "European_fire_salamander",
    26 to "common_newt",
    27 to "eft",
    28 to "spotted_salamander",
    29 to "axolotl",
    30 to "bullfrog",
    31 to "tree_frog",
    32 to "tailed_frog",
    33 to "loggerhead",
    34 to "leatherback_turtle",
    35 to "mud_turtle",
    36 to "terrapin",
    37 to "box_turtle",
    38 to "banded_gecko",
    39 to "common_iguana",
    40 to "American_chameleon",
    41 to "whiptail",
    42 to "agama",
    43 to "frilled_lizard",
    44 to "alligator_lizard",
    45 to "Gila_monster",
    46 to "green_lizard",
    47 to "African_chameleon",
    48 to "Komodo_dragon",
    49 to "African_crocodile",
    50 to "American_alligator",
    51 to "triceratops",
    52 to "thunder_snake",
    53 to "ringneck_snake",
    54 to "hognose_snake",
    55 to "green_snake",
    56 to "king_snake",
    57 to "garter_snake",
    58 to "water_snake",
    59 to "vine_snake",
    60 to "night_snake",
    61 to "boa_constrictor",
    62 to "rock_python",
    63 to "Indian_cobra",
    64 to "green_mamba",
    65 to "sea_snake",
    66 to "horned_viper",
    67 to "diamondback",
    68 to "sidewinder",
    69 to "trilobite",
    70 to "harvestman",
    71 to "scorpion",
    72 to "black_and_gold_garden_spider",
    73 to "barn_spider",
    74 to "garden_spider",
    75 to "black_widow",
    76 to "tarantula",
    77 to "wolf_spider",
    78 to "tick",
    79 to "centipede",
    80 to "black_grouse",
    81 to "ptarmigan",
    82 to "ruffed_grouse",
    83 to "prairie_chicken",
    84 to "peacock",
    85 to "quail",
    86 to "partridge",
    87 to "African_grey",
    88 to "macaw",
    89 to "sulphur-crested_cockatoo",
    90 to "lorikeet",
    91 to "coucal",
    92 to "bee_eater",
    93 to "hornbill",
    94 to "hummingbird",
    95 to "jacamar",
    96 to "toucan",
    97 to "drake",
    98 to "red-breasted_merganser",
    99 to "goose",
    100 to "black_swan",
    101 to "tusker",
    102 to "echidna",
    103 to "platypus",
    104 to "wallaby",
    105 to "koala",
    106 to "wombat",
    107 to "jellyfish",
    108 to "sea_anemone",
    109 to "brain_coral",
    110 to "flatworm",
    111 to "nematode",
    112 to "conch",
    113 to "snail",
    114 to "slug",
    115 to "sea_slug",
    116 to "chiton",
    117 to "chambered_nautilus",
    118 to "Dungeness_crab",
    119 to "rock_crab",
    120 to "fiddler_crab",
    121 to "king_crab",
    122 to "American_lobster",
    123 to "spiny_lobster",
    124 to "crayfish",
    125 to "hermit_crab",
    126 to "isopod",
    127 to "white_stork",
    128 to "black_stork",
    129 to "spoonbill",
    130 to "flamingo",
    131 to "little_blue_heron",
    132 to "American_egret",
    133 to "bittern",
    134 to "crane",
    135 to "limpkin",
    136 to "European_gallinule",
    137 to "American_coot",
    138 to "bustard",
    139 to "ruddy_turnstone",
    140 to "red-backed_sandpiper",
    141 to "redshank",
    142 to "dowitcher",
    143 to "oystercatcher",
    144 to "pelican",
    145 to "king_penguin",
    146 to "albatross",
    147 to "grey_whale",
    148 to "killer_whale",
    149 to "dugong",
    150 to "sea_lion",
    151 to "Chihuahua",
    152 to "Japanese_spaniel",
    153 to "Maltese_dog",
    154 to "Pekinese",
    155 to "Shih-Tzu",
    156 to "Blenheim_spaniel",
    157 to "papillon",
    158 to "toy_terrier",
    159 to "Rhodesian_ridgeback",
    160 to "Afghan_hound",
    161 to "basset",
    162 to "beagle",
    163 to "bloodhound",
    164 to "bluetick",
    165 to "black-and-tan_coonhound",
    166 to "Walker_hound",
    167 to "English_foxhound",
    168 to "redbone",
    169 to "borzoi",
    170 to "Irish_wolfhound",
    171 to "Italian_greyhound",
    172 to "whippet",
    173 to "Ibizan_hound",
    174 to "Norwegian_elkhound",
    175 to "otterhound",
    176 to "Saluki",
    177 to "Scottish_deerhound",
    178 to "Weimaraner",
    179 to "Staffordshire_bullterrier",
    180 to "American_Staffordshire_terrier",
    181 to "Bedlington_terrier",
    182 to "Border_terrier",
    183 to "Kerry_blue_terrier",
    184 to "Irish_terrier",
    185 to "Norfolk_terrier",
    186 to "Norwich_terrier",
    187 to "Yorkshire_terrier",
    188 to "wire-haired_fox_terrier",
    189 to "Lakeland_terrier",
    190 to "Sealyham_terrier",
    191 to "Airedale",
    192 to "cairn",
    193 to "Australian_terrier",
    194 to "Dandie_Dinmont",
    195 to "Boston_bull",
    196 to "miniature_schnauzer",
    197 to "giant_schnauzer",
    198 to "standard_schnauzer",
    199 to "Scotch_terrier",
    200 to "Tibetan_terrier",
    201 to "silky_terrier",
    202 to "soft-coated_wheaten_terrier",
    203 to "West_Highland_white_terrier",
    204 to "Lhasa",
    205 to "flat-coated_retriever",
    206 to "curly-coated_retriever",
    207 to "golden_retriever",
    208 to "Labrador_retriever",
    209 to "Chesapeake_Bay_retriever",
    210 to "German_short-haired_pointer",
    211 to "vizsla",
    212 to "English_setter",
    213 to "Irish_setter",
    214 to "Gordon_setter",
    215 to "Brittany_spaniel",
    216 to "clumber",
    217 to "English_springer",
    218 to "Welsh_springer_spaniel",
    219 to "cocker_spaniel",
    220 to "Sussex_spaniel",
    221 to "Irish_water_spaniel",
    222 to "kuvasz",
    223 to "schipperke",
    224 to "groenendael",
    225 to "malinois",
    226 to "briard",
    227 to "kelpie",
    228 to "komondor",
    229 to "Old_English_sheepdog",
    230 to "Shetland_sheepdog",
    231 to "collie",
    232 to "Border_collie",
    233 to "Bouvier_des_Flandres",
    234 to "Rottweiler",
    235 to "German_shepherd",
    236 to "Doberman",
    237 to "miniature_pinscher",
    238 to "Greater_Swiss_Mountain_dog",
    239 to "Bernese_mountain_dog",
    240 to "Appenzeller",
    241 to "EntleBucher",
    242 to "boxer",
    243 to "bull_mastiff",
    244 to "Tibetan_mastiff",
    245 to "French_bulldog",
    246 to "Great_Dane",
    247 to "Saint_Bernard",
    248 to "Eskimo_dog",
    249 to "malamute",
    250 to "Siberian_husky",
    251 to "dalmatian",
    252 to "affenpinscher",
    253 to "basenji",
    254 to "pug",
    255 to "Leonberg",
    256 to "Newfoundland",
    257 to "Great_Pyrenees",
    258 to "Samoyed",
    259 to "Pomeranian",
    260 to "chow",
    261 to "keeshond",
    262 to "Brabancon_griffon",
    263 to "Pembroke",
    264 to "Cardigan",
    265 to "toy_poodle",
    266 to "miniature_poodle",
    267 to "standard_poodle",
    268 to "Mexican_hairless",
    269 to "timber_wolf",
    270 to "white_wolf",
    271 to "red_wolf",
    272 to "coyote",
    273 to "dingo",
    274 to "dhole",
    275 to "African_hunting_dog",
    276 to "hyena",
    277 to "red_fox",
    278 to "kit_fox",
    279 to "Arctic_fox",
    280 to "grey_fox",
    281 to "tabby",
    282 to "tiger_cat",
    283 to "Persian_cat",
    284 to "Siamese_cat",
    285 to "Egyptian_cat",
    286 to "cougar",
    287 to "lynx",
    288 to "leopard",
    289 to "snow_leopard",
    290 to "jaguar",
    291 to "lion",
    292 to "tiger",
    293 to "cheetah",
    294 to "brown_bear",
    295 to "American_black_bear",
    296 to "ice_bear",
    297 to "sloth_bear",
    298 to "mongoose",
    299 to "meerkat",
    300 to "tiger_beetle",
    301 to "ladybug",
    302 to "ground_beetle",
    303 to "long-horned_beetle",
    304 to "leaf_beetle",
    305 to "dung_beetle",
    306 to "rhinoceros_beetle",
    307 to "weevil",
    308 to "fly",
    309 to "bee",
    310 to "ant",
    311 to "grasshopper",
    312 to "cricket",
    313 to "walking_stick",
    314 to "cockroach",
    315 to "mantis",
    316 to "cicada",
    317 to "leafhopper",
    318 to "lacewing",
    319 to "dragonfly",
    320 to "damselfly",
    321 to "admiral",
    322 to "ringlet",
    323 to "monarch",
    324 to "cabbage_butterfly",
    325 to "sulphur_butterfly",
    326 to "lycaenid",
    327 to "starfish",
    328 to "sea_urchin",
    329 to "sea_cucumber",
    330 to "wood_rabbit",
    331 to "hare",
    332 to "Angora",
    333 to "hamster",
    334 to "porcupine",
    335 to "fox_squirrel",
    336 to "marmot",
    337 to "beaver",
    338 to "guinea_pig",
    339 to "sorrel",
    340 to "zebra",
    341 to "hog",
    342 to "wild_boar",
    343 to "warthog",
    344 to "hippopotamus",
    345 to "ox",
    346 to "water_buffalo",
    347 to "bison",
    348 to "ram",
    349 to "bighorn",
    350 to "ibex",
    351 to "hartebeest",
    352 to "impala",
    353 to "gazelle",
    354 to "Arabian_camel",
    355 to "llama",
    356 to "weasel",
    357 to "mink",
    358 to "polecat",
    359 to "black-footed_ferret",
    360 to "otter",
    361 to "skunk",
    362 to "badger",
    363 to "armadillo",
    364 to "three-toed_sloth",
    365 to "orangutan",
    366 to "gorilla",
    367 to "chimpanzee",
    368 to "gibbon",
    369 to "siamang",
    370 to "guenon",
    371 to "patas",
    372 to "baboon",
    373 to "macaque",
    374 to "langur",
    375 to "colobus",
    376 to "proboscis_monkey",
    377 to "marmoset",
    378 to "capuchin",
    379 to "howler_monkey",
    380 to "titi",
    381 to "spider_monkey",
    382 to "squirrel_monkey",
    383 to "Madagascar_cat",
    384 to "indri",
    385 to "Indian_elephant",
    386 to "African_elephant",
    387 to "lesser_panda",
    388 to "giant_panda",
    389 to "barracouta",
    390 to "eel",
    391 to "coho",
    392 to "rock_beauty",
    393 to "anemone_fish",
    394 to "sturgeon",
    395 to "gar",
    396 to "lionfish",
    397 to "puffer",
    398 to "abacus",
    399 to "abaya",
    400 to "academic_gown",
    401 to "accordion",
    402 to "acoustic_guitar",
    403 to "aircraft_carrier",
    404 to "airliner",
    405 to "airship",
    406 to "altar",
    407 to "ambulance",
    408 to "amphibian",
    409 to "analog_clock",
    410 to "apiary",
    411 to "apron",
    412 to "ashcan",
    413 to "assault_rifle",
    414 to "backpack",
    415 to "bakery",
    416 to "balance_beam",
    417 to "balloon",
    418 to "ballpoint",
    419 to "Band_Aid",
    420 to "banjo",
    421 to "bannister",
    422 to "barbell",
    423 to "barber_chair",
    424 to "barbershop",
    425 to "barn",
    426 to "barometer",
    427 to "barrel",
    428 to "barrow",
    429 to "baseball",
    430 to "basketball",
    431 to "bassinet",
    432 to "bassoon",
    433 to "bathing_cap",
    434 to "bath_towel",
    435 to "bathtub",
    436 to "beach_wagon",
    437 to "beacon",
    438 to "beaker",
    439 to "bearskin",
    440 to "beer_bottle",
    441 to "beer_glass",
    442 to "bell_cote",
    443 to "bib",
    444 to "bicycle-built-for-two",
    445 to "bikini",
    446 to "binder",
    447 to "binoculars",
    448 to "birdhouse",
    449 to "boathouse",
    450 to "bobsled",
    451 to "bolo_tie",
    452 to "bonnet",
    453 to "bookcase",
    454 to "bookshop",
    455 to "bottlecap",
    456 to "bow",
    457 to "bow_tie",
    458 to "brass",
    459 to "brassiere",
    460 to "breakwater",
    461 to "breastplate",
    462 to "broom",
    463 to "bucket",
    464 to "buckle",
    465 to "bulletproof_vest",
    466 to "bullet_train",
    467 to "butcher_shop",
    468 to "cab",
    469 to "caldron",
    470 to "candle",
    471 to "cannon",
    472 to "canoe",
    473 to "can_opener",
    474 to "cardigan",
    475 to "car_mirror",
    476 to "carousel",
    477 to "carpenter's_kit",
    478 to "carton",
    479 to "car_wheel",
    480 to "cash_machine",
    481 to "cassette",
    482 to "cassette_player",
    483 to "castle",
    484 to "catamaran",
    485 to "CD_player",
    486 to "cello",
    487 to "cellular_telephone",
    488 to "chain",
    489 to "chainlink_fence",
    490 to "chain_mail",
    491 to "chain_saw",
    492 to "chest",
    493 to "chiffonier",
    494 to "chime",
    495 to "china_cabinet",
    496 to "Christmas_stocking",
    497 to "church",
    498 to "cinema",
    499 to "cleaver",
    500 to "cliff_dwelling",
    501 to "cloak",
    502 to "clog",
    503 to "cocktail_shaker",
    504 to "coffee_mug",
    505 to "coffeepot",
    506 to "coil",
    507 to "combination_lock",
    508 to "computer_keyboard",
    509 to "confectionery",
    510 to "container_ship",
    511 to "convertible",
    512 to "corkscrew",
    513 to "cornet",
    514 to "cowboy_boot",
    515 to "cowboy_hat",
    516 to "cradle",
    517 to "crane",
    518 to "crash_helmet",
    519 to "crate",
    520 to "crib",
    521 to "Crock_Pot",
    522 to "croquet_ball",
    523 to "crutch",
    524 to "cuirass",
    525 to "dam",
    526 to "desk",
    527 to "desktop_computer",
    528 to "dial_telephone",
    529 to "diaper",
    530 to "digital_clock",
    531 to "digital_watch",
    532 to "dining_table",
    533 to "dishrag",
    534 to "dishwasher",
    535 to "disk_brake",
    536 to "dock",
    537 to "dogsled",
    538 to "dome",
    539 to "doormat",
    540 to "drilling_platform",
    541 to "drum",
    542 to "drumstick",
    543 to "dumbbell",
    544 to "Dutch_oven",
    545 to "electric_fan",
    546 to "electric_guitar",
    547 to "electric_locomotive",
    548 to "entertainment_center",
    549 to "envelope",
    550 to "espresso_maker",
    551 to "face_powder",
    552 to "feather_boa",
    553 to "file",
    554 to "fireboat",
    555 to "fire_engine",
    556 to "fire_screen",
    557 to "flagpole",
    558 to "flute",
    559 to "folding_chair",
    560 to "football_helmet",
    561 to "forklift",
    562 to "fountain",
    563 to "fountain_pen",
    564 to "four-poster",
    565 to "freight_car",
    566 to "French_horn",
    567 to "frying_pan",
    568 to "fur_coat",
    569 to "garbage_truck",
    570 to "gasmask",
    571 to "gas_pump",
    572 to "goblet",
    573 to "go-kart",
    574 to "golf_ball",
    575 to "golfcart",
    576 to "gondola",
    577 to "gong",
    578 to "gown",
    579 to "grand_piano",
    580 to "greenhouse",
    581 to "grille",
    582 to "grocery_store",
    583 to "guillotine",
    584 to "hair_slide",
    585 to "hair_spray",
    586 to "half_track",
    587 to "hammer",
    588 to "hamper",
    589 to "hand_blower",
    590 to "hand-held_computer",
    591 to "handkerchief",
    592 to "hard_disc",
    593 to "harmonica",
    594 to "harp",
    595 to "harvester",
    596 to "hatchet",
    597 to "holster",
    598 to "home_theater",
    599 to "honeycomb",
    600 to "hook",
    601 to "hoopskirt",
    602 to "horizontal_bar",
    603 to "horse_cart",
    604 to "hourglass",
    605 to "iPod",
    606 to "iron",
    607 to "jack-o'-lantern",
    608 to "jean",
    609 to "jeep",
    610 to "jersey",
    611 to "jigsaw_puzzle",
    612 to "jinrikisha",
    613 to "joystick",
    614 to "kimono",
    615 to "knee_pad",
    616 to "knot",
    617 to "lab_coat",
    618 to "ladle",
    619 to "lampshade",
    620 to "laptop",
    621 to "lawn_mower",
    622 to "lens_cap",
    623 to "letter_opener",
    624 to "library",
    625 to "lifeboat",
    626 to "lighter",
    627 to "limousine",
    628 to "liner",
    629 to "lipstick",
    630 to "Loafer",
    631 to "lotion",
    632 to "loudspeaker",
    633 to "loupe",
    634 to "lumbermill",
    635 to "magnetic_compass",
    636 to "mailbag",
    637 to "mailbox",
    638 to "maillot",
    639 to "maillot",
    640 to "manhole_cover",
    641 to "maraca",
    642 to "marimba",
    643 to "mask",
    644 to "matchstick",
    645 to "maypole",
    646 to "maze",
    647 to "measuring_cup",
    648 to "medicine_chest",
    649 to "megalith",
    650 to "microphone",
    651 to "microwave",
    652 to "military_uniform",
    653 to "milk_can",
    654 to "minibus",
    655 to "miniskirt",
    656 to "minivan",
    657 to "missile",
    658 to "mitten",
    659 to "mixing_bowl",
    660 to "mobile_home",
    661 to "Model_T",
    662 to "modem",
    663 to "monastery",
    664 to "monitor",
    665 to "moped",
    666 to "mortar",
    667 to "mortarboard",
    668 to "mosque",
    669 to "mosquito_net",
    670 to "motor_scooter",
    671 to "mountain_bike",
    672 to "mountain_tent",
    673 to "mouse",
    674 to "mousetrap",
    675 to "moving_van",
    676 to "muzzle",
    677 to "nail",
    678 to "neck_brace",
    679 to "necklace",
    680 to "nipple",
    681 to "notebook",
    682 to "obelisk",
    683 to "oboe",
    684 to "ocarina",
    685 to "odometer",
    686 to "oil_filter",
    687 to "organ",
    688 to "oscilloscope",
    689 to "overskirt",
    690 to "oxcart",
    691 to "oxygen_mask",
    692 to "packet",
    693 to "paddle",
    694 to "paddlewheel",
    695 to "padlock",
    696 to "paintbrush",
    697 to "pajama",
    698 to "palace",
    699 to "panpipe",
    700 to "paper_towel",
    701 to "parachute",
    702 to "parallel_bars",
    703 to "park_bench",
    704 to "parking_meter",
    705 to "passenger_car",
    706 to "patio",
    707 to "pay-phone",
    708 to "pedestal",
    709 to "pencil_box",
    710 to "pencil_sharpener",
    711 to "perfume",
    712 to "Petri_dish",
    713 to "photocopier",
    714 to "pick",
    715 to "pickelhaube",
    716 to "picket_fence",
    717 to "pickup",
    718 to "pier",
    719 to "piggy_bank",
    720 to "pill_bottle",
    721 to "pillow",
    722 to "ping-pong_ball",
    723 to "pinwheel",
    724 to "pirate",
    725 to "pitcher",
    726 to "plane",
    727 to "planetarium",
    728 to "plastic_bag",
    729 to "plate_rack",
    730 to "plow",
    731 to "plunger",
    732 to "Polaroid_camera",
    733 to "pole",
    734 to "police_van",
    735 to "poncho",
    736 to "pool_table",
    737 to "pop_bottle",
    738 to "pot",
    739 to "potter's_wheel",
    740 to "power_drill",
    741 to "prayer_rug",
    742 to "printer",
    743 to "prison",
    744 to "projectile",
    745 to "projector",
    746 to "puck",
    747 to "punching_bag",
    748 to "purse",
    749 to "quill",
    750 to "quilt",
    751 to "racer",
    752 to "racket",
    753 to "radiator",
    754 to "radio",
    755 to "radio_telescope",
    756 to "rain_barrel",
    757 to "recreational_vehicle",
    758 to "reel",
    759 to "reflex_camera",
    760 to "refrigerator",
    761 to "remote_control",
    762 to "restaurant",
    763 to "revolver",
    764 to "rifle",
    765 to "rocking_chair",
    766 to "rotisserie",
    767 to "rubber_eraser",
    768 to "rugby_ball",
    769 to "rule",
    770 to "running_shoe",
    771 to "safe",
    772 to "safety_pin",
    773 to "saltshaker",
    774 to "sandal",
    775 to "sarong",
    776 to "sax",
    777 to "scabbard",
    778 to "scale",
    779 to "school_bus",
    780 to "schooner",
    781 to "scoreboard",
    782 to "screen",
    783 to "screw",
    784 to "screwdriver",
    785 to "seat_belt",
    786 to "sewing_machine",
    787 to "shield",
    788 to "shoe_shop",
    789 to "shoji",
    790 to "shopping_basket",
    791 to "shopping_cart",
    792 to "shovel",
    793 to "shower_cap",
    794 to "shower_curtain",
    795 to "ski",
    796 to "ski_mask",
    797 to "sleeping_bag",
    798 to "slide_rule",
    799 to "sliding_door",
    800 to "slot",
    801 to "snorkel",
    802 to "snowmobile",
    803 to "snowplow",
    804 to "soap_dispenser",
    805 to "soccer_ball",
    806 to "sock",
    807 to "solar_dish",
    808 to "sombrero",
    809 to "soup_bowl",
    810 to "space_bar",
    811 to "space_heater",
    812 to "space_shuttle",
    813 to "spatula",
    814 to "speedboat",
    815 to "spider_web",
    816 to "spindle",
    817 to "sports_car",
    818 to "spotlight",
    819 to "stage",
    820 to "steam_locomotive",
    821 to "steel_arch_bridge",
    822 to "steel_drum",
    823 to "stethoscope",
    824 to "stole",
    825 to "stone_wall",
    826 to "stopwatch",
    827 to "stove",
    828 to "strainer",
    829 to "streetcar",
    830 to "stretcher",
    831 to "studio_couch",
    832 to "stupa",
    833 to "submarine",
    834 to "suit",
    835 to "sundial",
    836 to "sunglass",
    837 to "sunglasses",
    838 to "sunscreen",
    839 to "suspension_bridge",
    840 to "swab",
    841 to "sweatshirt",
    842 to "swimming_trunks",
    843 to "swing",
    844 to "switch",
    845 to "syringe",
    846 to "table_lamp",
    847 to "tank",
    848 to "tape_player",
    849 to "teapot",
    850 to "teddy",
    851 to "television",
    852 to "tennis_ball",
    853 to "thatch",
    854 to "theater_curtain",
    855 to "thimble",
    856 to "thresher",
    857 to "throne",
    858 to "tile_roof",
    859 to "toaster",
    860 to "tobacco_shop",
    861 to "toilet_seat",
    862 to "torch",
    863 to "totem_pole",
    864 to "tow_truck",
    865 to "toyshop",
    866 to "tractor",
    867 to "trailer_truck",
    868 to "tray",
    869 to "trench_coat",
    870 to "tricycle",
    871 to "trimaran",
    872 to "tripod",
    873 to "triumphal_arch",
    874 to "trolleybus",
    875 to "trombone",
    876 to "tub",
    877 to "turnstile",
    878 to "typewriter_keyboard",
    879 to "umbrella",
    880 to "unicycle",
    881 to "upright",
    882 to "vacuum",
    883 to "vase",
    884 to "vault",
    885 to "velvet",
    886 to "vending_machine",
    887 to "vestment",
    888 to "viaduct",
    889 to "violin",
    890 to "volleyball",
    891 to "waffle_iron",
    892 to "wall_clock",
    893 to "wallet",
    894 to "wardrobe",
    895 to "warplane",
    896 to "washbasin",
    897 to "washer",
    898 to "water_bottle",
    899 to "water_jug",
    900 to "water_tower",
    901 to "whiskey_jug",
    902 to "whistle",
    903 to "wig",
    904 to "window_screen",
    905 to "window_shade",
    906 to "Windsor_tie",
    907 to "wine_bottle",
    908 to "wing",
    909 to "wok",
    910 to "wooden_spoon",
    911 to "wool",
    912 to "worm_fence",
    913 to "wreck",
    914 to "yawl",
    915 to "yurt",
    916 to "web_site",
    917 to "comic_book",
    918 to "crossword_puzzle",
    919 to "street_sign",
    920 to "traffic_light",
    921 to "book_jacket",
    922 to "menu",
    923 to "plate",
    924 to "guacamole",
    925 to "consomme",
    926 to "hot_pot",
    927 to "trifle",
    928 to "ice_cream",
    929 to "ice_lolly",
    930 to "French_loaf",
    931 to "bagel",
    932 to "pretzel",
    933 to "cheeseburger",
    934 to "hotdog",
    935 to "mashed_potato",
    936 to "head_cabbage",
    937 to "broccoli",
    938 to "cauliflower",
    939 to "zucchini",
    940 to "spaghetti_squash",
    941 to "acorn_squash",
    942 to "butternut_squash",
    943 to "cucumber",
    944 to "artichoke",
    945 to "bell_pepper",
    946 to "cardoon",
    947 to "mushroom",
    948 to "Granny_Smith",
    949 to "strawberry",
    950 to "orange",
    951 to "lemon",
    952 to "fig",
    953 to "pineapple",
    954 to "banana",
    955 to "jackfruit",
    956 to "custard_apple",
    957 to "pomegranate",
    958 to "hay",
    959 to "carbonara",
    960 to "chocolate_sauce",
    961 to "dough",
    962 to "meat_loaf",
    963 to "pizza",
    964 to "potpie",
    965 to "burrito",
    966 to "red_wine",
    967 to "espresso",
    968 to "cup",
    969 to "eggnog",
    970 to "alp",
    971 to "bubble",
    972 to "cliff",
    973 to "coral_reef",
    974 to "geyser",
    975 to "lakeside",
    976 to "promontory",
    977 to "sandbar",
    978 to "seashore",
    979 to "valley",
    980 to "volcano",
    981 to "ballplayer",
    982 to "groom",
    983 to "scuba_diver",
    984 to "rapeseed",
    985 to "daisy",
    986 to "yellow_lady's_slipper",
    987 to "corn",
    988 to "acorn",
    989 to "hip",
    990 to "buckeye",
    991 to "coral_fungus",
    992 to "agaric",
    993 to "gyromitra",
    994 to "stinkhorn",
    995 to "earthstar",
    996 to "hen-of-the-woods",
    997 to "bolete",
    998 to "ear",
    999 to "toilet_tissue"
)