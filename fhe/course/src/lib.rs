//! The course: public physics constants and the 18 holes, read from the same
//! `course.json` the web front end renders, and their compiled form (wall
//! faces, terrain gradient) that the simulation uses.
//!
//! A hole is a convex boundary of rails, convex blocks and round bumpers inside
//! it, water ponds (which end the shot), and one quadratic terrain.

use std::sync::OnceLock;

use serde::Deserialize;

/// The course as authored, shared with the web front end.
pub const COURSE_JSON: &str = include_str!("../../../web/src/lib/golf/course.json");

pub type Vec2 = [f64; 2];

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Physics {
    /// m/s².
    pub gravity: f64,
    /// Fraction of gravity that drives a rolling ball (5/7 for a solid sphere).
    pub roll_factor: f64,
    /// Linear drag λ, 1/s.
    pub drag: f64,
    pub ball_radius: f64,
    /// Time grid of the simulation, s.
    pub dt: f64,
    /// How long each segment between two contacts is followed, s.
    pub horizon: f64,
    /// Contacts the simulation resolves per shot.
    pub max_contacts: usize,
    /// Fastest shot, m/s.
    pub max_speed: f64,
    pub stop_speed: f64,
    pub wall_rest_speed: f64,
    pub cup_radius: f64,
    pub capture_speed: f64,
    pub max_strokes: u32,
    pub restitution: Restitution,
}

impl Physics {
    /// Steps of the time grid in one segment.
    pub fn steps(&self) -> usize {
        (self.horizon / self.dt).round() as usize
    }

    /// Acceleration of gravity on a rolling ball, per unit of slope.
    pub fn roll(&self) -> f64 {
        self.gravity * self.roll_factor
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Restitution {
    pub rail: f64,
    pub block: f64,
    pub bumper: f64,
}

/// Height h(q) = ½ dᵀHd + s·d with d = q − center.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Terrain {
    pub center: Vec2,
    /// H = [[xx, xy], [xy, yy]], 1/m.
    pub hessian: [f64; 3],
    /// Gradient at the center.
    pub slope: Vec2,
}

impl Terrain {
    pub const FLAT: Terrain = Terrain { center: [0.0, 0.0], hessian: [0.0; 3], slope: [0.0, 0.0] };

    pub fn height(&self, x: f64, y: f64) -> f64 {
        let (dx, dy) = (x - self.center[0], y - self.center[1]);
        let [hxx, hxy, hyy] = self.hessian;
        0.5 * (hxx * dx * dx + 2.0 * hxy * dx * dy + hyy * dy * dy) + self.slope[0] * dx + self.slope[1] * dy
    }

    pub fn gradient(&self, x: f64, y: f64) -> Vec2 {
        let (dx, dy) = (x - self.center[0], y - self.center[1]);
        let [hxx, hxy, hyy] = self.hessian;
        [hxx * dx + hxy * dy + self.slope[0], hxy * dx + hyy * dy + self.slope[1]]
    }
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ObstacleSpec {
    Block { points: Vec<Vec2>, restitution: Option<f64> },
    Bumper { center: Vec2, radius: f64, restitution: Option<f64> },
}

#[derive(Deserialize, Debug, Clone)]
pub struct HoleSpec {
    pub name: String,
    pub par: u32,
    pub tee: Vec2,
    pub cup: Vec2,
    /// Counter-clockwise, convex.
    pub boundary: Vec<Vec2>,
    pub terrain: Option<Terrain>,
    #[serde(default)]
    pub obstacles: Vec<ObstacleSpec>,
    /// Convex ponds.
    #[serde(default)]
    pub water: Vec<Vec<Vec2>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct CourseSpec {
    pub physics: Physics,
    pub holes: Vec<HoleSpec>,
}

/// A straight face: `clearance(q) = n·q − b` is the ball center's distance to
/// it, minus the ball radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub n: Vec2,
    pub b: f64,
}

impl Face {
    pub fn clearance(&self, x: f64, y: f64) -> f64 {
        self.n[0] * x + self.n[1] * y - self.b
    }
}

#[derive(Debug, Clone)]
pub enum Obstacle {
    /// The boundary, inward normals: the ball touches it when any face's clearance is negative.
    Rail { faces: Vec<Face>, restitution: f64 },
    /// Outward normals: the ball touches it when every face's clearance is negative.
    Block { faces: Vec<Face>, restitution: f64 },
}

impl Obstacle {
    /// Whether a ball centered at (x, y) overlaps the obstacle.
    pub fn touches(&self, x: f64, y: f64) -> bool {
        match self {
            Obstacle::Rail { faces, .. } => faces.iter().any(|f| f.clearance(x, y) < 0.0),
            Obstacle::Block { faces, .. } => faces.iter().all(|f| f.clearance(x, y) < 0.0),
        }
    }
}

/// Faces the encrypted simulation tests at each time step, at most: it packs
/// 256 steps × 32 faces into a quarter of a ciphertext's slots.
pub const MAX_FACES: usize = 32;

/// What the simulation tests for contact: each rail face on its own, and
/// each block (bumpers included). The ball is inside a detector when it is
/// behind all of its faces.
#[derive(Debug, Clone, PartialEq)]
pub struct Detector {
    /// Its faces, `faces[first..first + count]` of the hole.
    pub first: usize,
    pub count: usize,
    pub rail: bool,
    pub restitution: f64,
    /// What a bounce off two faces or more of the block at once is multiplied
    /// by: `1/λ`, `λ` the largest eigenvalue of `Σ n·nᵀ` over the runs of
    /// faces a point outside can be in front of together (at least 1). Each
    /// face then reflects its share of the velocity and the bounce never adds
    /// speed; 1 for blocks whose adjacent faces meet at right angles.
    pub multi_face_scale: f64,
}

/// The `multi_face_scale` of a convex block of `faces`, counter-clockwise, and
/// the most faces a point outside can be in front of together.
pub fn front_runs(faces: &[Face]) -> (f64, usize) {
    let n = faces.len();
    let (mut largest, mut longest) = (1.0f64, 1usize);
    for start in 0..n {
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for len in 1..n {
            let f = &faces[(start + len - 1) % n];
            // A point is in front of a run of faces only if their normals lie in
            // an open half-plane: the run turns by less than half a turn.
            let first = &faces[start];
            let turn = (first.n[0] * f.n[1] - first.n[1] * f.n[0]).atan2(first.n[0] * f.n[0] + first.n[1] * f.n[1]);
            if len > 1 && turn.rem_euclid(std::f64::consts::TAU) >= std::f64::consts::PI - 1e-9 {
                break;
            }
            longest = longest.max(len);
            sxx += f.n[0] * f.n[0];
            sxy += f.n[0] * f.n[1];
            syy += f.n[1] * f.n[1];
            if len >= 2 {
                let (m, d) = ((sxx + syy) / 2.0, ((sxx - syy) / 2.0).hypot(sxy));
                largest = largest.max(m + d);
            }
        }
    }
    (1.0 / largest, longest)
}

impl Detector {
    pub fn faces<'a>(&self, faces: &'a [Face]) -> &'a [Face] {
        &faces[self.first..self.first + self.count]
    }

    pub fn inside(&self, faces: &[Face], x: f64, y: f64) -> bool {
        self.faces(faces).iter().all(|f| f.clearance(x, y) < 0.0)
    }
}

#[derive(Debug, Clone)]
pub struct Hole {
    pub index: usize,
    pub spec: HoleSpec,
    pub terrain: Terrain,
    /// The rail first, then blocks and bumpers.
    pub obstacles: Vec<Obstacle>,
    /// Every face the simulation tests, by detector.
    pub faces: Vec<Face>,
    pub detectors: Vec<Detector>,
}

/// The regular octagon with its corners on the circle of `radius` around `c`,
/// counter-clockwise, its faces facing the axes and the diagonals.
pub fn octagon(c: Vec2, radius: f64) -> Vec<Vec2> {
    (0..8)
        .map(|i| {
            let a = (i as f64 + 0.5) * std::f64::consts::FRAC_PI_4;
            [c[0] + radius * a.cos(), c[1] + radius * a.sin()]
        })
        .collect()
}

fn detectors(obstacles: &[Obstacle]) -> (Vec<Face>, Vec<Detector>) {
    let (mut faces, mut detectors) = (Vec::new(), Vec::new());
    for o in obstacles {
        match o {
            Obstacle::Rail { faces: fs, restitution } => {
                for f in fs {
                    detectors.push(Detector {
                        first: faces.len(),
                        count: 1,
                        rail: true,
                        restitution: *restitution,
                        multi_face_scale: 1.0,
                    });
                    faces.push(*f);
                }
            }
            Obstacle::Block { faces: fs, restitution } => {
                detectors.push(Detector {
                    first: faces.len(),
                    count: fs.len(),
                    rail: false,
                    restitution: *restitution,
                    multi_face_scale: front_runs(fs).0,
                });
                faces.extend(fs);
            }
        }
    }
    (faces, detectors)
}

#[derive(Debug, Clone)]
pub struct Course {
    pub physics: Physics,
    pub holes: Vec<Hole>,
    /// FNV-1a of `COURSE_JSON`, to check that the client and the server agree.
    pub hash: u64,
}

fn signed_area2(p: &[Vec2]) -> f64 {
    (0..p.len())
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % p.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum()
}

fn is_convex_ccw(p: &[Vec2]) -> bool {
    p.len() >= 3
        && signed_area2(p) > 0.0
        && (0..p.len()).all(|i| {
            let (a, b, c) = (p[i], p[(i + 1) % p.len()], p[(i + 2) % p.len()]);
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0
        })
}

/// The faces of a convex counter-clockwise polygon, offset by the ball radius.
fn polygon_faces(p: &[Vec2], inward: bool, radius: f64) -> Vec<Face> {
    (0..p.len())
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % p.len()]);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len = ex.hypot(ey);
            // Left of a counter-clockwise edge is inside.
            let sign = if inward { 1.0 } else { -1.0 };
            let n = [-ey / len * sign, ex / len * sign];
            Face { n, b: n[0] * a[0] + n[1] * a[1] + radius }
        })
        .collect()
}

/// Whether the point is strictly inside a convex counter-clockwise polygon.
pub fn inside_polygon(p: &[Vec2], x: f64, y: f64) -> bool {
    (0..p.len()).all(|i| {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) > 0.0
    })
}

fn compile_hole(spec: HoleSpec, index: usize, physics: &Physics) -> Result<Hole, String> {
    let r = physics.ball_radius;
    let at = format!("hole {} ({})", index + 1, spec.name);
    if !is_convex_ccw(&spec.boundary) {
        return Err(format!("{at}: the boundary is not convex counter-clockwise"));
    }
    let mut obstacles =
        vec![Obstacle::Rail { faces: polygon_faces(&spec.boundary, true, r), restitution: physics.restitution.rail }];
    for o in &spec.obstacles {
        obstacles.push(match o {
            ObstacleSpec::Block { points, restitution } => {
                if !is_convex_ccw(points) {
                    return Err(format!("{at}: a block is not convex counter-clockwise"));
                }
                Obstacle::Block {
                    faces: polygon_faces(points, false, r),
                    restitution: restitution.unwrap_or(physics.restitution.block),
                }
            }
            // An octagonal post: a block with the bumper's restitution.
            ObstacleSpec::Bumper { center, radius, restitution } => Obstacle::Block {
                faces: polygon_faces(&octagon(*center, *radius), false, r),
                restitution: restitution.unwrap_or(physics.restitution.bumper),
            },
        });
    }
    if spec.water.iter().any(|w| !is_convex_ccw(w)) {
        return Err(format!("{at}: a pond is not convex counter-clockwise"));
    }
    for (label, [x, y]) in [("tee", spec.tee), ("cup", spec.cup)] {
        if obstacles.iter().any(|o| o.touches(x, y)) {
            return Err(format!("{at}: the {label} is not on the green"));
        }
        if spec.water.iter().any(|w| inside_polygon(w, x, y)) {
            return Err(format!("{at}: the {label} is in water"));
        }
    }
    let (faces, detectors) = detectors(&obstacles);
    if faces.len() > MAX_FACES {
        return Err(format!("{at}: {} faces to test, the simulation handles {MAX_FACES}", faces.len()));
    }
    let terrain = spec.terrain.clone().unwrap_or(Terrain::FLAT);
    Ok(Hole { index, spec, terrain, obstacles, faces, detectors })
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

impl Course {
    pub fn parse(json: &str) -> Result<Course, String> {
        let spec: CourseSpec = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let holes = spec
            .holes
            .into_iter()
            .enumerate()
            .map(|(i, h)| compile_hole(h, i, &spec.physics))
            .collect::<Result<_, _>>()?;
        Ok(Course { physics: spec.physics, holes, hash: fnv1a(json.as_bytes()) })
    }

    /// The course built into this binary.
    pub fn get() -> &'static Course {
        static COURSE: OnceLock<Course> = OnceLock::new();
        COURSE.get_or_init(|| Course::parse(COURSE_JSON).expect("invalid course.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_course_has_18_holes() {
        let course = Course::get();
        assert_eq!(course.holes.len(), 18);
        assert!(course.holes.iter().all(|h| matches!(h.obstacles[0], Obstacle::Rail { .. })));
    }

    #[test]
    fn multi_face_bounces_are_scaled_off_right_angles_only() {
        let square = polygon_faces(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], false, 0.0);
        assert_eq!(front_runs(&square).1, 2);
        assert!((front_runs(&square).0 - 1.0).abs() < 1e-12);
        // An octagon's faces facing a point span up to 135°: `Σ n·nᵀ` reaches 2.
        let post = polygon_faces(&octagon([0.0, 0.0], 0.2), false, 0.0);
        assert_eq!(front_runs(&post).1, 4);
        assert!((front_runs(&post).0 - 0.5).abs() < 1e-9, "{:?}", front_runs(&post));
        let course = Course::get();
        let scales: Vec<f64> =
            course.holes.iter().flat_map(|h| h.detectors.iter().map(|d| d.multi_face_scale)).collect();
        assert!(scales.iter().all(|&s| s > 0.0 && s <= 1.0));
    }

    #[test]
    fn faces_measure_clearance_inward() {
        let faces = polygon_faces(&[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], true, 0.1);
        // At (0.5, 0.5): 0.5 from the bottom, 1.5 from the right, 0.5 from the top and left.
        let c: Vec<f64> = faces.iter().map(|f| f.clearance(0.5, 0.5)).collect();
        for (got, want) in c.iter().zip([0.4, 1.4, 0.4, 0.4]) {
            assert!((got - want).abs() < 1e-12, "{c:?}");
        }
    }

    #[test]
    fn rejects_a_clockwise_boundary() {
        let json =
            COURSE_JSON.replacen("[[0, 0], [1.2, 0], [1.2, 6], [0, 6]]", "[[0, 0], [0, 6], [1.2, 6], [1.2, 0]]", 1);
        assert!(Course::parse(&json).unwrap_err().contains("convex"));
    }
}
