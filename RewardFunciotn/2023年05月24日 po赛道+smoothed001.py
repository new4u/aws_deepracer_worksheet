# Source URL: https://github.com/dp770/aws_deepracer_worksheet/blob/main/src/reward_function.py
import math

# constants
MAX_SPEED = 4.0
MAX_STEERING = 30.0
MAX_DIRECTION_DIFF = 30.0
MAX_STEPS_TO_DECAY_PENALTY = 0      # Value of zero or below disables penalty for having wheels off track
MAX_STEPS_TO_PROGRESS_RATIO = 1.8   # Desired maximum number of steps to be taken for 1% of progress
RACING_LINE_SMOOTHING_STEPS = 2
RACING_LINE_WIDTH_FREE_ZONE = 0.10  # Percentage of racing line width for 100% of "being on track" reward
RACING_LINE_WIDTH_SAFE_ZONE = 0.35  # Percentage of racing line width for distance relative "being on track" reward
RACING_LINE_VS_CENTRAL_LINE = 0.90  # Number in range of [0, 1]. Zero forces to follow central line, 1 - racing line
SENSITIVITY_EXP_CNT_DISTANCE = 3.00  # Higher number gives more freedom on the track, can cause zig-zags
SENSITIVITY_EXP_ACTION_SPEED = 3.00  # Higher number increases penalty for low speed
SENSITIVITY_EXP_ACTION_STEER = 0.70  # Higher number decreases penalty for high steering
SENSITIVITY_EXP_DIR_STEERING = 2.00  # Lower number accelerates penalty increase for not following track direction
TOTAL_PENALTY_ON_OFF_TRACK = 0.999999  # Maximum penalty in percentage of total reward for being off track
TOTAL_PENALTY_ON_OFF_DIR_STEER = 0.35  # Maximum penalty in percentage of total reward for off directional steering
TOTAL_PENALTY_ON_HIGH_STEERING = 0.15  # Maximum penalty in percentage of total reward for high steering
REWARD_WEIGHT_PROG_STEP = 30
REWARD_WEIGHT_MAX_SPEED = 25
REWARD_WEIGHT_MIN_STEER = 20
REWARD_WEIGHT_DIR_STEER = 15
REWARD_WEIGHT_ON_TRACK = 10
MAX_TOTAL_REWARD = REWARD_WEIGHT_ON_TRACK + REWARD_WEIGHT_PROG_STEP + REWARD_WEIGHT_DIR_STEER + \
                   REWARD_WEIGHT_MAX_SPEED + REWARD_WEIGHT_MIN_STEER

# static
smoothed_central_line= [[-0.388003206413952, -4.294770264160247], [-0.1048284967441955, -4.286602028843021], [0.17947694730971364, -4.2802808473896], [0.4649621460778763, -4.275443248971349], [0.7516731845686202, -4.271768659151095], [1.039642752763573, -4.268972630242381], [1.3288846877416207, -4.2668078274673915], [1.6193878027692843, -4.265056230243859], [1.9111122628969588, -4.263521647820845], [2.2039872471576833, -4.262019578731863], [2.4979102165086156, -4.260366776041263], [2.792747161992726, -4.258370281577327], [3.0883334217450065, -4.2558167534211835], [3.3844746456313817, -4.252462471720374], [3.6809475592658796, -4.248024577655923], [3.9775006790507477, -4.242174525776735], [4.273854767932874, -4.2345338934668], [4.569704752289259, -4.224675426582125], [4.864721273630135, -4.2121261293754335], [5.158558899571121, -4.1963830615132665], [5.450859601063209, -4.176922965346763], [5.741277037165706, -4.153244950051669], [6.029478149964812, -4.124877638517249], [6.315192765794755, -4.091457987274596], [6.59821881777772, -4.052745823656788], [6.878445656966303, -4.008644015501117], [7.156024203987072, -3.959503183849433], [7.430861082748784, -3.9052212887136943], [7.7044251677009035, -3.84834574509381], [7.974865051019941, -3.7856826986411116], [8.246209314226942, -3.724401300268868], [8.513450424887312, -3.6551629803583965], [8.761061699921932, -3.5573322017008198], [8.991363761500953, -3.438238579680977], [9.195300332395727, -3.290556447291377], [9.352673209092647, -3.105202452299977], [9.459990901230029, -2.8912817237549673], [9.52825359185247, -2.6616606448032623], [9.555469649438235, -2.420713058931563], [9.543175988640394, -2.175143895327334], [9.51169257439474, -1.929628332956991], [9.455316087873612, -1.6882361049532142], [9.394363379601222, -1.444599478439853], [9.253885042702565, -1.2288783606567257], [9.1083600120866, -1.018101308932921], [8.926123842160896, -0.8262904514952448], [8.717576416889676, -0.6559035970537296], [8.520769715393579, -0.4749818338446629], [8.31743091292308, -0.29842061159088534], [8.11624408112796, -0.11913583724925596], [7.914235942830057, 0.060545227127134414], [7.711922925007211, 0.24056899602777454], [7.511231555465292, 0.4230973931646464], [7.3074007335415985, 0.6024758990241097], [7.1109228288771655, 0.7900251600527941], [6.897339648761559, 0.9587574925553516], [6.715772738773333, 1.1622045800245087], [6.563980418335535, 1.389312148999903], [6.425167929843499, 1.624315950401477], [6.2804211290843295, 1.853348104070574], [6.1402868365385785, 2.083873937711049], [6.001604408397835, 2.313231678512111], [5.8662240443799165, 2.542496081256105], [5.7333848237024565, 2.77084568214269], [5.603253822001297, 2.9982129210981974], [5.47527700107438, 3.2240362424197153], [5.3493196837441905, 3.4482269385096633], [5.224470172346938, 3.6700496013643287], [5.101303110512115, 3.890329444164638], [4.97750616881488, 4.107093053954577], [4.857267180493684, 4.325020013058284], [4.731532150985927, 4.535381593723615], [4.618847174544798, 4.757555194549849], [4.481572653384829, 4.953811978581481], [4.3209720712519015, 5.113482274806206], [4.133779895553402, 5.208818207601058], [3.9336958721406745, 5.24245036813477], [3.735267661872278, 5.181091037625353], [3.559149041067919, 5.043552570075541], [3.410692798856655, 4.847928274236407], [3.286400289836118, 4.616272207341148], [3.1766167791072535, 4.366438698744123], [3.059975023049608, 4.123817589933447], [2.9462928121992085, 3.876874422922109], [2.83095899933065, 3.6307003697192823], [2.7161359735185786, 3.382669847574246], [2.6006480038882094, 3.1340072109750308], [2.4847307333370616, 2.884397943788473], [2.3678500296446927, 2.6344404259665413], [2.2497919430998916, 2.3844371609750645], [2.130156450188564, 2.134930395461757], [2.0088017215329925, 1.8862443234445763], [1.8853898636936681, 1.6389042439691996], [1.7603104988904446, 1.3927603789351641], [1.6328917587964265, 1.1486257746116653], [1.5055131923256573, 0.904395686512725], [1.3736500916832615, 0.6647657723623144], [1.2475986581267082, 0.4196481078920361], [1.1069670206581543, 0.1897289391747227], [0.9279196922115827, 0.007821228088277819], [0.7048640301419536, -0.09983022409335857], [0.45857170970226, -0.14846188970641844], [0.20235431147855562, -0.16106157773417912], [-0.05777377664840609, -0.14666115277920322], [-0.31544700449528984, -0.14447300324530776], [-0.5744333443070413, -0.136264696352327], [-0.8322948583623552, -0.12823956300757], [-1.09366229232558, -0.1268649372346149], [-1.3478094643158987, -0.09639752129803278], [-1.5968030470141348, -0.052207567666956195], [-1.8321346801094573, 0.019795930621303727], [-2.0392537317534187, 0.1369666362845131], [-2.1917812203012756, 0.3167171078707507], [-2.273759351647576, 0.5552632358611617], [-2.284671941498028, 0.8358825110725377], [-2.329544785186353, 1.0982949000003743], [-2.3588187773574267, 1.3711664347210142], [-2.396060347517139, 1.6412712645894456], [-2.430600154605474, 1.914724704909563], [-2.4682884046322786, 2.188323768863081], [-2.506860260937709, 2.463340925581024], [-2.548247660687595, 2.7386901307264653], [-2.5923799428259686, 3.014338262126206], [-2.6403349598307924, 3.2895831885943663], [-2.692537380100422, 3.56405374780095], [-2.749780917119967, 3.8371351344571467], [-2.8121668103578625, 4.10862624911195], [-2.880490894340223, 4.377855217082006], [-2.953426144148549, 4.645540089572009], [-3.0332184752497744, 4.91004427617658], [-3.1133389008925394, 5.175537789472475], [-3.2052731419843035, 5.434268194816673], [-3.2832581662289497, 5.703153790514973], [-3.3994300752403674, 5.947117243280265], [-3.530615797756327, 6.178197949367995], [-3.681999687965634, 6.391675832393277], [-3.853595855988191, 6.58219650587994], [-4.039019752998825, 6.754843068179793], [-4.248221946174216, 6.886945341146645], [-4.472379560349908, 6.981084871229572], [-4.706675587860877, 7.03569103234927], [-4.9466690271077285, 7.046567133767237], [-5.185727679560938, 7.013925720335445], [-5.419181925713909, 6.940817401728577], [-5.638961145002481, 6.820867723003063], [-5.846812952460264, 6.674775446214273], [-6.040618414341002, 6.5022455134915536], [-6.216440680331735, 6.302717506648845], [-6.3767678647953385, 6.084231387688353], [-6.523315969978157, 5.850359640817923], [-6.653339523415818, 5.600991154615858], [-6.788853826307511, 5.356757481221379], [-6.9191250553781485, 5.107551038746147], [-7.04718426616997, 4.8561009874969026], [-7.170925270121976, 4.600480802164208], [-7.290689010263252, 4.340993268476321], [-7.40613713013041, 4.077383785103847], [-7.517426922774608, 3.809845061500474], [-7.624774419496189, 3.5386497762300007], [-7.728511675809434, 3.264169835089223], [-7.829027653850291, 2.9868293639850694], [-7.926724974713238, 2.7070553288558417], [-8.02199879863482, 2.425261354144737], [-8.115208064388321, 2.141822509923889], [-8.206662992942105, 1.8570693186261977], [-8.296612292732645, 1.5712821393909993], [-8.385236953046517, 1.284693100318956], [-8.472645376297141, 0.9974898695696223], [-8.558870625622792, 0.7098219537594438], [-8.6438682616207, 0.42180788148422316], [-8.727514612442127, 0.13354298866850758], [-8.809605263707482, -0.15489268550157487], [-8.88985399754645, -0.4434265914931159], [-8.967892842353596, -0.731986211743597], [-9.043274052555997, -1.0204921401291172], [-9.11547547104073, -1.3088522270491487], [-9.183911119102303, -1.5969571761702435], [-9.247947218015769, -1.8846784554232738], [-9.306932125290057, -2.171870015319279], [-9.360223522396144, -2.4583720498470942], [-9.407260320993446, -2.7440262446406996], [-9.447583207327966, -3.0286815643230467], [-9.480975706307955, -3.312236701457692], [-9.507522088205297, -3.5946385014004756], [-9.527474892090654, -3.8759244284948786], [-9.542209576104435, -4.156271342768826], [-9.551495326866652, -4.435823812904318], [-9.564212896312211, -4.716052385407836], [-9.561813418007985, -4.993880794520755], [-9.535210498076983, -5.264951892273874], [-9.492116655907218, -5.530181375795271], [-9.425656201081043, -5.785574975290048], [-9.331711099657023, -6.026502824094178], [-9.211163398476488, -6.250113859805235], [-9.055629990572768, -6.446289168391231], [-8.865797712790915, -6.606806069771848], [-8.655069610895673, -6.737004064224289], [-8.434400011576969, -6.850259985547168], [-8.20341761667897, -6.93926032002028], [-7.966433631865409, -7.013029673889928], [-7.72157781397956, -7.046434351631189], [-7.4752778940076725, -7.049417310878824], [-7.230971029555667, -7.010985420308255], [-6.994998071903341, -6.92921084298573], [-6.7735514488103705, -6.794296769250733], [-6.577798555484845, -6.6061333692973845], [-6.367494803298697, -6.44591897495904], [-6.162570366049163, -6.2728383989103165], [-5.953383176439895, -6.1060096599031635], [-5.744831323400437, -5.93576999551549], [-5.534014596948659, -5.76773533278282], [-5.323226317768669, -5.597920320308033], [-5.109817485951626, -5.430973476598537], [-4.8985533451229415, -5.259410731415271], [-4.680991104049155, -5.096941195050673], [-4.476626505097929, -4.912688566258539], [-4.242308469794733, -4.777275955348389], [-3.9871805729882124, -4.68323477570516], [-3.7077576754847104, -4.6531912703036715], [-3.438831130989162, -4.595668784608674], [-3.164032796518774, -4.554882136025337], [-2.8913738729431766, -4.5096622489177225], [-2.616567240088478, -4.47159198308796], [-2.341586520778888, -4.435561990859393], [-2.065347843554014, -4.4044846212131255], [-1.7883155449385892, -4.377071440511848], [-1.5102279921528137, -4.353859219605611], [-1.2311946183639257, -4.3342992569144485], [-0.9511481237481716, -4.318220268488556], [-0.6700984095173567, -4.305167923879789]]

was_off_track_at_step = -MAX_STEPS_TO_DECAY_PENALTY
previous_steps_reward = MAX_TOTAL_REWARD

# Range [-180:+180]
def calc_slope(prev_point, next_point):
    return math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))


# Range [0:180]
def calc_direction_diff(steering, heading, track_direction):
    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = steering + heading - track_direction
    if direction_diff > 180.0:
        direction_diff = direction_diff - 360.0
    if direction_diff < -180.0:
        direction_diff = direction_diff + 360.0
    return abs(direction_diff)


# Returns distance between two points in meters
def calc_distance(prev_point, next_point):
    delta_x = next_point[0] - prev_point[0]
    delta_y = next_point[1] - prev_point[1]
    return math.sqrt(delta_x * delta_x + delta_y * delta_y)


def smooth_central_line(center_line, max_offset, pp=0.10, p=0.05, c=0.70, n=0.05, nn=0.10, iterations=72, skip_step=1):
    smoothed_line = center_line
    for i in range(0, iterations):
        smoothed_line = smooth_central_line_internal(center_line, max_offset, smoothed_line, pp, p, c, n, nn, skip_step)
    return smoothed_line


def smooth_central_line_internal(center_line, max_offset, smoothed_line, pp, p, c, n, nn, skip_step):
    length = len(center_line)
    new_line = [[0.0 for _ in range(2)] for _ in range(length)]
    for i in range(0, length):
        wpp = smoothed_line[(i - 2 * skip_step + length) % length]
        wp = smoothed_line[(i - skip_step + length) % length]
        wc = smoothed_line[i]
        wn = smoothed_line[(i + skip_step) % length]
        wnn = smoothed_line[(i + 2 * skip_step) % length]
        new_line[i][0] = pp * wpp[0] + p * wp[0] + c * wc[0] + n * wn[0] + nn * wnn[0]
        new_line[i][1] = pp * wpp[1] + p * wp[1] + c * wc[1] + n * wn[1] + nn * wnn[1]
        while calc_distance(new_line[i], center_line[i]) >= max_offset:
            new_line[i][0] = (0.98 * new_line[i][0]) + (0.02 * center_line[i][0])
            new_line[i][1] = (0.98 * new_line[i][1]) + (0.02 * center_line[i][1])
    return new_line


# Calculate distance between current point and closest point on line between prev_point and next_point
def calc_distance_from_line(curr_point, prev_point, next_point):
    distance_cp_to_pp = calc_distance(curr_point, prev_point)  # b
    distance_cp_to_np = calc_distance(curr_point, next_point)  # a
    distance_pp_to_np = calc_distance(prev_point, next_point)  # c
    # cos A = (b^2 + c^2 - a^2) / 2bc
    angle_pp = math.acos((distance_cp_to_pp * distance_cp_to_pp + distance_pp_to_np * distance_pp_to_np
                          - distance_cp_to_np * distance_cp_to_np) / (2 * distance_cp_to_pp * distance_pp_to_np))
    # b / sin(Pi/2) = d / sin(A)
    return distance_cp_to_pp * math.sin(angle_pp)


def ema(prev, new, period):
    k = 2.0 / (1.0 + period)
    return (new - prev) * k + prev


# Reward function expected by AWS DeepRacer API
def reward_function(params):
    track_width = params['track_width']
    waypoints = params['waypoints']
    # initialize central line
    global smoothed_central_line
    if smoothed_central_line is None:
        max_offset = track_width * RACING_LINE_VS_CENTRAL_LINE * 0.5
        smoothed_central_line = smooth_central_line(waypoints, max_offset, skip_step=RACING_LINE_SMOOTHING_STEPS)
        print("track_waypoints:", "track_width =", track_width,
              "\ntrack_original =", waypoints, "\ntrack_smoothed =", smoothed_central_line)

    # re-initialize was_off_track_at_step
    global was_off_track_at_step
    steps = params['steps']
    if steps < was_off_track_at_step:
        was_off_track_at_step = -MAX_STEPS_TO_DECAY_PENALTY
    if not params['all_wheels_on_track']:
        was_off_track_at_step = steps

    global previous_steps_reward
    if steps <= 2:
        previous_steps_reward = MAX_TOTAL_REWARD

    # Calculate penalty for wheels being or have recently been off track
    wheels_off_track_penalty = 1.0
    if MAX_STEPS_TO_DECAY_PENALTY > 0:
        wheels_off_track_penalty = min(steps - was_off_track_at_step, MAX_STEPS_TO_DECAY_PENALTY) / (
            1.0 * MAX_STEPS_TO_DECAY_PENALTY)

    # Speed penalty to keep the car moving fast
    speed = params['speed']  # Range: 0.0:4.0
    speed_ratio = speed / MAX_SPEED
    reward_max_speed = REWARD_WEIGHT_MAX_SPEED * pow(speed_ratio, SENSITIVITY_EXP_ACTION_SPEED)

    # Steering penalty to discourage zig-zags
    steering = params['steering_angle']  # Range: -30:30
    steering_ratio = abs(steering / MAX_STEERING)
    reward_min_steering = REWARD_WEIGHT_MIN_STEER * (1.0 - pow(steering_ratio, SENSITIVITY_EXP_ACTION_STEER))

    # Reward on directional move to the next milestone
    wp_length = len(smoothed_central_line)
    wp_indices = params['closest_waypoints']
    curr_point = [params['x'], params['y']]
    prev_point = smoothed_central_line[wp_indices[0]]
    next_point_1 = smoothed_central_line[(wp_indices[1] + 1) % wp_length]
    next_point_2 = smoothed_central_line[(wp_indices[1] + 2) % wp_length]
    next_point_3 = smoothed_central_line[(wp_indices[1] + 3) % wp_length]
    track_direction_1 = calc_slope(prev_point, next_point_1)
    track_direction_2 = calc_slope(prev_point, next_point_2)
    track_direction_3 = calc_slope(prev_point, next_point_3)

    heading = params['heading']  # Range: -180:+180
    direction_diff_ratio = (
            0.20 * min((calc_direction_diff(steering, heading, track_direction_1) / MAX_DIRECTION_DIFF), 1.00) +
            0.30 * min((calc_direction_diff(steering, heading, track_direction_2) / MAX_DIRECTION_DIFF), 1.00) +
            0.50 * min((calc_direction_diff(steering, heading, track_direction_3) / MAX_DIRECTION_DIFF), 1.00))
    dir_steering_ratio = 1.0 - pow(direction_diff_ratio, SENSITIVITY_EXP_DIR_STEERING)
    reward_dir_steering = REWARD_WEIGHT_DIR_STEER * dir_steering_ratio

    # Reward on close distance to the racing line
    free_zone = track_width * RACING_LINE_WIDTH_FREE_ZONE * 0.5
    safe_zone = track_width * RACING_LINE_WIDTH_SAFE_ZONE * 0.5
    dislocation = calc_distance_from_line(curr_point, prev_point, next_point_1)
    on_track_ratio = 0.0
    if dislocation <= free_zone:
        on_track_ratio = 1.0
    elif dislocation <= safe_zone:
        on_track_ratio = 1.0 - pow(dislocation / safe_zone, SENSITIVITY_EXP_CNT_DISTANCE)
    reward_on_track = on_track_ratio * REWARD_WEIGHT_ON_TRACK

    # Reward on good progress per step
    progress = params['progress']
    reward_prog_step = REWARD_WEIGHT_PROG_STEP * min(1.0, MAX_STEPS_TO_PROGRESS_RATIO * (progress / steps))

    reward_total = reward_on_track + reward_max_speed + reward_min_steering + reward_dir_steering + reward_prog_step
    reward_total -= reward_total * (1.0 - on_track_ratio) * TOTAL_PENALTY_ON_OFF_TRACK
    reward_total -= reward_total * (1.0 - dir_steering_ratio) * TOTAL_PENALTY_ON_OFF_DIR_STEER
    reward_total -= reward_total * steering_ratio * TOTAL_PENALTY_ON_HIGH_STEERING
    reward_total *= wheels_off_track_penalty

    print("rewards:" + (20 * "{:.4f}," + "{:.4f}").format(reward_total, wheels_off_track_penalty,
        reward_on_track, reward_max_speed, reward_min_steering, reward_dir_steering, reward_prog_step,
        dislocation, track_direction_1, track_direction_2, track_direction_3, direction_diff_ratio,
        waypoints[wp_indices[0]][0], waypoints[wp_indices[0]][1], prev_point[0], prev_point[1],
        next_point_1[0], next_point_1[1], next_point_2[0], next_point_2[1], next_point_3[0], next_point_3[1]))

    # previous_steps_reward = ema(previous_steps_reward, reward_total, 3)
    return float(0.0000001 + reward_total)