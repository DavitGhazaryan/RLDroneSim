-- Reset to (N,E,AGL) via COMMAND_LONG MAV_CMD_USER_1
local ZERO = Vector3f(); ZERO:x(0); ZERO:y(0); ZERO:z(0)
local origin, last_seq = nil, -1

local mavlink_msgs = require("MAVLink.mavlink_msgs")
local COMMAND_LONG_ID = mavlink_msgs.get_msgid("COMMAND_LONG")
local CMD_USER1       = 31010 -- MAV_CMD_USER_1

mavlink:init(10, 1)
mavlink:register_rx_msgid(COMMAND_LONG_ID)
mavlink:block_command(CMD_USER1)  -- prevent core from sending UNSUPPORTED acks

local function m_per_deg_lon(lat_deg) return 111319.5*math.cos(math.rad(lat_deg)) end

local function capture_origin_once()
  if origin then return true end
  local h = ahrs:get_home(); if not h then return false end
  origin = Location(); origin:lat(h:lat()); origin:lng(h:lng()); origin:alt(h:alt())
  return true
end

local function target_from_offsets(n_m, e_m, agl_m)
  local lat = origin:lat()/1e7; local lon = origin:lng()/1e7
  local lat_o = math.floor((lat + n_m/111319.5)*1e7 + 0.5)
  local lon_o = math.floor((lon + e_m/m_per_deg_lon(lat))*1e7 + 0.5)
  local alt_cm = origin:alt() + math.floor(agl_m*100 + 0.5)
  local L=Location(); L:lat(lat_o); L:lng(lon_o); L:alt(alt_cm); return L
end

local function do_reset(n_m, e_m, agl_m)
  if not capture_origin_once() then return end
  local tgt = target_from_offsets(n_m, e_m, agl_m)
  if ahrs.reset then ahrs:reset() end
  sim:set_pose(0, tgt, Quaternion(), ZERO, ZERO)

end

function update()
  capture_origin_once()

  local msg, chan = mavlink:receive_chan()
  if msg then
    local parsed = mavlink_msgs.decode(msg, { [COMMAND_LONG_ID] = "COMMAND_LONG" })
    if parsed and parsed.msgid == COMMAND_LONG_ID and parsed.command == CMD_USER1 then
      local n   = parsed.param1 or 0
      local e   = parsed.param2 or 0
      local agl = parsed.param3 or 0
      local seq = math.floor(parsed.param4 or -1)

      if seq ~= last_seq then
        do_reset(n, e, agl)
        last_seq = seq
      end

      -- ACK (ACCEPTED=0) back on same channel
      local ack = { command = CMD_USER1, result = 0, progress = 0, result_param2 = 0,
                    target_system = parsed.sysid, target_component = parsed.compid }
      mavlink:send_chan(chan, mavlink_msgs.encode("COMMAND_ACK", ack))
    end
  end

  return update, 50 -- 20 Hz
end

return update()
