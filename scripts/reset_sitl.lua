-- Reset to (N,E,AGL) relative to a fixed origin; SCR_USER1 edge-triggered.
-- SCR_USER1: trigger (0->1 runs once)
-- SCR_USER2: North  (m)
-- SCR_USER3: East   (m)
-- SCR_USER4: Alt AGL (m) relative to origin/home AMSL

local ZERO = Vector3f(); ZERO:x(0); ZERO:y(0); ZERO:z(0)
local origin = nil  -- fixed WGS84/AMSL captured once
local last_trig = 0

local function m_per_deg_lon(lat_deg) return 111319.5*math.cos(math.rad(lat_deg)) end

local function capture_origin_once()
  if origin then return true end
  local h = ahrs:get_home()
  if not h then return false end
  origin = Location(); origin:lat(h:lat()); origin:lng(h:lng()); origin:alt(h:alt())
  gcs:send_text(6,"Origin locked")
  return true
end

local function target_from_offsets(n_m, e_m, agl_m)
  local lat = origin:lat()/1e7
  local lon = origin:lng()/1e7
  local lat_o = math.floor( (lat + n_m/111319.5) *1e7 + 0.5 )
  local lon_o = math.floor( (lon + e_m/m_per_deg_lon(lat)) *1e7 + 0.5 )
  local alt_cm = origin:alt() + math.floor(agl_m*100 + 0.5) -- AMSL = origin AMSL + AGL
  local L=Location(); L:lat(lat_o); L:lng(lon_o); L:alt(alt_cm); return L
end

local function do_reset(n_m,e_m,agl_m)
  if not capture_origin_once() then return end
  local tgt = target_from_offsets(n_m,e_m,agl_m)
  local q = Quaternion() -- identity (keep simple for altitude work)

  -- SOFT estimator reset: keep origin/home; just reset EKF state
  if ahrs.reset then ahrs:reset() end

  -- Teleport truth, zero velocities
  sim:set_pose(0, tgt, q, ZERO, ZERO)

  -- Clear FCU targets
  vehicle:set_target_location(tgt)
  vehicle:set_target_velocity_NED(ZERO)

  gcs:send_text(6, string.format("Reset to N=%.2f E=%.2f AGL=%.2f (origin held)", n_m, e_m, agl_m))
end

local function loop()
  capture_origin_once()
  local trig = param:get("SCR_USER1") or 0
  if trig==1 and last_trig==0 then
    local N = param:get("SCR_USER2") or 0
    local E = param:get("SCR_USER3") or 0
    local Z = param:get("SCR_USER4") or 0
    do_reset(N,E,Z)
  end
  last_trig = trig
  return loop, 200
end

gcs:send_text(0,"Reset: SCR_USER1 edge; offsets N/E in USER2/3, AGL in USER4; origin fixed")
return loop, 200
