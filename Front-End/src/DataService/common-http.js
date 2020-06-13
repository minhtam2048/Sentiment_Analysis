import axios from "axios";

const httpCommon = (payload, method = "POST") => {
  console.log(payload);
  return axios({
    method,
    url: "http://localhost:8000/api/posts",
    headers: {
      "Content-Type": "application/json"
    },
    data: payload
  });
};

export default httpCommon;