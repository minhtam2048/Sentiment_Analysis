import commonHttp from "./common-http";

class DataService {
    postData(data) {
        return commonHttp.post("/posts", data)
    }
}

export default new DataService();