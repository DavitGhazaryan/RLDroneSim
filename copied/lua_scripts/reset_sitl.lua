-- -- Reset to (N,E,AGL) via COMMAND_LONG MAV_CMD_USER_1
-- local ZERO = Vector3f(); ZERO:x(0); ZERO:y(0); ZERO:z(0)
-- local last_seq = -1

-- local mavlink_msgs = require("MAVLink.mavlink_msgs")
-- local COMMAND_LONG_ID = mavlink_msgs.get_msgid("COMMAND_LONG")
-- local CMD_USER1       = 31010 -- MAV_CMD_USER_1

-- mavlink:init(10, 1)
-- mavlink:register_rx_msgid(COMMAND_LONG_ID)
-- mavlink:block_command(CMD_USER1)

-- local function finite(x) return x == x and x ~= math.huge and x ~= -math.huge end
-- local function to_cm(m) return math.floor(m * 100 + 0.5) end

-- local function get_origin()
--   local L = ahrs:get_origin()
--   if L then return L end
--   local h = ahrs:get_home()
--   if not h then return nil end
--   L = Location(); L:lat(h:lat()); L:lng(h:lng()); L:alt(h:alt()); return L
-- end

-- local function target_from_offsets(n_m, e_m, agl_m)
--   if not (finite(n_m) and finite(e_m) and finite(agl_m)) then return nil end
--   local O = get_origin(); if not O then return nil end
--   local L = Location(); L:lat(O:lat()); L:lng(O:lng()); L:alt(O:alt())
--   L:offset(n_m, e_m)                -- safe meters→lat/lon
--   L:alt(O:alt() + to_cm(agl_m))     -- cm
--   return L
-- end

-- local function reset_controller_state()
--     gcs:send_text(0, "Entered")
--     poscontrol:reset_controller()

-- end

-- local function do_reset(n_m, e_m, agl_m)
--   local tgt = target_from_offsets(n_m, e_m, agl_m)
--   if not tgt then gcs:send_text(0, "reset: no origin/invalid inputs"); return end

--   if ahrs.reset then ahrs:reset() end
-- --   gcs:send_text(0, "Here do reset")
--   reset_controller_state()
  
--   local q = Quaternion()
--   local yaw = ahrs:get_yaw_rad() or 0    -- radians
--   q:from_euler(0, 0, yaw)

--   sim:set_pose(0, tgt, q, ZERO, ZERO)
-- end

-- function update()
--   gcs:send_text(0, "Here")
--   local msg, chan = mavlink:receive_chan()
--   if msg then
--     local parsed = mavlink_msgs.decode(msg, { [COMMAND_LONG_ID]="COMMAND_LONG" })
--     if parsed and parsed.msgid == COMMAND_LONG_ID and parsed.command == CMD_USER1 then
--       local n   = tonumber(parsed.param1) or 0
--       local e   = tonumber(parsed.param2) or 0
--       local agl = tonumber(parsed.param3) or 0
--       local seq = math.floor(tonumber(parsed.param4) or -1)

--       if seq ~= last_seq then
--         -- do_reset(n, e, agl)
--         last_seq = seq
--       end

--       local ack = { command = CMD_USER1, result = 0, progress = 0, result_param2 = 0,
--                     target_system = parsed.sysid, target_component = parsed.compid }
--       mavlink:send_chan(chan, mavlink_msgs.encode("COMMAND_ACK", ack))
--     end
--   end
--   return update, 50 -- 20 Hz
-- end

-- return update()
